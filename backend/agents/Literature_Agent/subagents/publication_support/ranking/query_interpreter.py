"""
LLM query interpretation (Stage 0).

The retrieval pipeline is sensitive to phrasing: the raw user string seeds
both the BGE vector (embeddings.embedder) and the hierarchy label matching
(ranking.ranker), so a pasted abstract, a one-word query or a sentence
carrying constraints all degrade retrieval in different ways.

This module puts the LLM *in front of* the pipeline instead of only at
re-ranking time. It reads the raw user input and returns a structured
reading of it: a clean topic to embed, alternate phrasings, field hints and
any constraints the user expressed in prose.

It is strictly an enhancement layer. Every failure path falls back to the
raw input, so interpretation can never break retrieval.
"""

import os
import json
from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()


API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")


if not API_KEY:
    raise RuntimeError("AZURE_OPENAI_API_KEY is missing")

if not ENDPOINT:
    raise RuntimeError("AZURE_OPENAI_ENDPOINT is missing")

if not DEPLOYMENT:
    raise RuntimeError("AZURE_OPENAI_DEPLOYMENT is missing")


client = OpenAI(
    api_key=API_KEY,
    base_url=ENDPOINT,
)


# Interpreting the same input twice costs a call and returns the same thing,
# so results are memoised for the life of the process.
_interpretation_cache = {}


SYSTEM_PROMPT = """You are the query-understanding stage of an academic \
journal recommender. You read a researcher's raw input and turn it into a \
structured search intent. You never recommend journals yourself.

The `search_query` you produce is embedded by a BGE sentence-transformer and \
matched against journal scope descriptions and OpenAlex topic labels. This \
means it MUST be fluent, descriptive natural language — a phrase or sentence \
naming the research area, the way a human would describe it.

NEVER emit keyword soup. "deep learning, medical imaging, CNN, segmentation" \
is wrong. "Deep learning methods for medical image segmentation" is right.

Rules for `search_query`:
- If the input is already a clean topic, keep it essentially unchanged. Do \
not embellish a query that is already good.
- If the input is an abstract or a long paragraph, condense it to the core \
research area in one sentence.
- If the input is in another language, translate it to English.
- Strip publication constraints out of it (open access, impact factor, \
review speed, publisher preferences) — those belong in `constraints`, not in \
the text that gets embedded.

Return ONLY a JSON object with exactly these keys:

{
  "is_valid": true,
  "intent": "search",
  "search_query": "Clean descriptive research topic",
  "expanded_queries": ["Alternate phrasing", "Another angle on the topic"],
  "field_hints": ["Computer Science"],
  "constraints": {
    "open_access": null,
    "publisher_exclude": [],
    "other": []
  },
  "needs_clarification": false,
  "clarification_question": ""
}

Field meanings:
- is_valid: false ONLY for empty, nonsense or non-research input. A real but \
vague research area ("AI", "biology") is still valid — mark it valid and set \
needs_clarification instead.
- intent: one of "search", "followup", "chitchat", "unclear".
- expanded_queries: 1-2 genuinely different phrasings of the same topic \
(synonyms, adjacent terminology). Same natural-language rule applies. Empty \
list if the topic is too narrow to rephrase meaningfully.
- field_hints: broad academic fields, e.g. "Medicine", "Computer Science", \
"Materials Science". Empty list if unclear.
- constraints: open_access is true/false/null. publisher_exclude and other \
are lists of strings, empty when nothing was expressed.
- needs_clarification: true only when the topic is too vague to search at \
all (e.g. "AI", "biology"). If true, clarification_question must be a single \
specific question."""


# Shape and per-field defaults. Anything the model omits or mistypes is
# replaced from here rather than raising.
def _defaults(raw_input: str) -> dict:

    return {
        "is_valid": True,
        "intent": "search",
        "search_query": raw_input,
        "expanded_queries": [],
        "field_hints": [],
        "constraints": {
            "open_access": None,
            "publisher_exclude": [],
            "other": [],
        },
        "needs_clarification": False,
        "clarification_question": "",
        "raw_input": raw_input,
        "interpreted": False,
    }


def _strip_code_fences(text: str) -> str:

    if not text.startswith("```"):
        return text

    text = text.split("```")[1]

    if text.startswith("json"):
        text = text[4:]

    return text.strip()


def _coerce(parsed: dict, raw_input: str) -> dict:
    """Merge the model's JSON onto the defaults, keeping only sane values."""

    result = _defaults(raw_input)

    # A blank or non-string search_query would poison the embedding, so the
    # raw input stands in whenever the model gives us nothing usable.
    search_query = parsed.get("search_query")

    if isinstance(search_query, str) and search_query.strip():
        result["search_query"] = search_query.strip()

    if isinstance(parsed.get("is_valid"), bool):
        result["is_valid"] = parsed["is_valid"]

    if parsed.get("intent") in ("search", "followup", "chitchat", "unclear"):
        result["intent"] = parsed["intent"]

    for key in ("expanded_queries", "field_hints"):

        value = parsed.get(key)

        if isinstance(value, list):
            result[key] = [
                item.strip()
                for item in value
                if isinstance(item, str) and item.strip()
            ]

    constraints = parsed.get("constraints")

    if isinstance(constraints, dict):

        if isinstance(constraints.get("open_access"), bool):
            result["constraints"]["open_access"] = constraints["open_access"]

        for key in ("publisher_exclude", "other"):

            value = constraints.get(key)

            if isinstance(value, list):
                result["constraints"][key] = [
                    item.strip()
                    for item in value
                    if isinstance(item, str) and item.strip()
                ]

    if isinstance(parsed.get("needs_clarification"), bool):
        result["needs_clarification"] = parsed["needs_clarification"]

    question = parsed.get("clarification_question")

    if isinstance(question, str):
        result["clarification_question"] = question.strip()

    # Never ask a clarifying question we don't actually have.
    if result["needs_clarification"] and not result["clarification_question"]:
        result["needs_clarification"] = False

    result["interpreted"] = True

    return result


def _call_model(raw_input: str) -> str:

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": raw_input},
    ]

    # JSON mode when the deployment supports it; some do not, so a plain
    # call is retried before giving up and falling back to the raw input.
    try:
        response = client.chat.completions.create(
            model=DEPLOYMENT,
            messages=messages,
            response_format={"type": "json_object"},
            temperature=0,
            max_tokens=400,
        )

    except Exception:
        response = client.chat.completions.create(
            model=DEPLOYMENT,
            messages=messages,
            temperature=0,
            max_tokens=400,
        )

    return response.choices[0].message.content.strip()


def interpret_query(raw_input: str, use_cache: bool = True) -> dict:
    """
    Read raw user input and return a structured search intent.

    Args:
        raw_input: Whatever the user typed — a topic, an abstract, a
                   sentence with constraints in it.
        use_cache: Reuse a previous interpretation of the same input.

    Returns:
        Dict with the keys described in SYSTEM_PROMPT, plus `raw_input` and
        `interpreted` (False when the LLM was unavailable and the raw input
        is being passed through untouched).

        `search_query` is always a non-empty string safe to embed.
    """

    if not raw_input or not raw_input.strip():

        result = _defaults("")
        result["is_valid"] = False
        result["intent"] = "unclear"

        return result

    raw_input = raw_input.strip()

    if use_cache and raw_input in _interpretation_cache:
        return _interpretation_cache[raw_input]

    try:
        response_text = _strip_code_fences(_call_model(raw_input))

        parsed = json.loads(response_text)

        if not isinstance(parsed, dict):
            raise ValueError("Expected a JSON object")

        result = _coerce(parsed, raw_input)

    except json.JSONDecodeError as error:
        print(f"[interpreter] Could not parse LLM response: {error}")
        result = _defaults(raw_input)

    except Exception as error:
        print(f"[interpreter] Falling back to raw query: {error}")
        result = _defaults(raw_input)

    if use_cache:
        _interpretation_cache[raw_input] = result

    return result


def describe_interpretation(interpretation: dict) -> str:
    """One-line human summary, for CLI output."""

    if not interpretation["interpreted"]:
        return "Using your input as-is (interpreter unavailable)."

    parts = [f"Searching for: {interpretation['search_query']}"]

    if interpretation["field_hints"]:
        parts.append(
            "Fields: " + ", ".join(interpretation["field_hints"])
        )

    constraints = interpretation["constraints"]

    expressed = []

    if constraints["open_access"] is True:
        expressed.append("open access only")

    if constraints["open_access"] is False:
        expressed.append("open access not required")

    if constraints["publisher_exclude"]:
        expressed.append(
            "excluding " + ", ".join(constraints["publisher_exclude"])
        )

    expressed.extend(constraints["other"])

    if expressed:
        parts.append("Noted: " + "; ".join(expressed))

    return "\n".join(parts)


def build_llm_topic(interpretation: dict) -> str:
    """
    Topic string handed to the re-ranker.

    Retrieval gets the cleaned query, but the re-ranker is judging fit to the
    researcher's actual work, so it sees the original wording and any
    constraints the interpreter lifted out of it as well.
    """

    raw = interpretation["raw_input"]
    clean = interpretation["search_query"]

    if not interpretation["interpreted"] or raw == clean:
        return clean

    # Publication preferences are deliberately left out: the re-ranker is
    # asked to judge topical fit only, and ranking.llm_reranker applies the
    # constraints afterwards in code. Restating them here would work against
    # that instruction. The original wording still goes through, since it
    # carries intent the cleaned query drops.
    return "\n".join([
        clean,
        f"\nResearcher's original words: {raw}",
    ])


if __name__ == "__main__":

    samples = [
        "machine learning for medical imaging",
        "I need an open access journal for my work on CRISPR gene editing "
        "in plants, but not anything from Elsevier",
        "AI",
        "apprentissage automatique pour l'imagerie médicale",
    ]

    for sample in samples:

        print("=" * 60)
        print(f"Input: {sample}\n")

        interpretation = interpret_query(sample)

        print(json.dumps(interpretation, indent=2))
        print()
