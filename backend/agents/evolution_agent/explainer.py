"""Explainer (LLM #2) for the Evolution Agent.

Turns the structured result produced by the selected sub-agent(s) into a
short, grounded, human-readable interpretation.

Contract
--------
* It is the SECOND and LAST LLM call: Planner + Explainer = 2 per request.
* It uses the same client / deployment as the Planner
  (``framework.llm_client.get_llm``) — no second provider, no extra .env.
* It receives ONLY the whitelisted fields listed in ``explain``: never the
  environment, credentials, the orchestrator, the full shared context, a
  worker object, or the output of a worker that was not selected.
* It returns ONLY a string.  The structured payload (scores, groups,
  network, Newick, bootstrap, model, confidence) is produced exclusively by
  the workers and is never rebuilt from the model's answer.
* It never raises.  ``None`` means "no trustworthy interpretation": the
  caller keeps the worker result, keeps ``status=completed``, and records a
  ``interpretation_unavailable`` warning.
"""

from __future__ import annotations

import json
import logging
import re

_logger = logging.getLogger(__name__)

# Fields the orchestrator is allowed to hand over. Anything else is a bug.
ALLOWED_INPUT_KEYS = frozenset({
    "instruction", "feature", "species", "results", "warnings",
    "providers_are_mocked",
})

_MAX_CHARS = 900

# Ordinary English words that legitimately follow a genus name in prose.
# Without this list "the Mus+Gallus node has bootstrap 100" was read as a
# hallucinated species "Gallus node" and the whole answer was discarded.
_PROSE_AFTER_GENUS = frozenset({
    "node", "nodes", "clade", "clades", "branch", "branches", "lineage",
    "lineages", "group", "groups", "cluster", "clusters", "split", "splits",
    "taxon", "taxa", "leaf", "leaves", "tip", "tips", "sequence", "sequences",
    "sample", "samples", "pair", "pairs", "subtree", "tree", "trees",
    "and", "with", "has", "have", "had", "was", "were", "are", "is", "being",
    "appears", "appear", "forms", "form", "shows", "show", "sits", "sit",
    "groups", "clusters", "falls", "fall", "shares", "share", "than", "then",
    "versus", "vs", "against", "from", "into", "onto", "near", "next",
    "similarity", "support", "supports", "confidence", "score", "scores",
    "bootstrap", "model", "topology", "position", "placement", "relative",
    "together", "separately", "alone", "also", "still", "however", "which",
    "that", "this", "these", "those", "both", "each", "either", "neither",
    "protein", "proteins", "genome", "genomes", "gene", "genes",
})

EXPLAINER_SYSTEM_PROMPT = """\
You are the explainer of the Evolution Agent (Umbrella BioHub).

You receive a user's request and the STRUCTURED RESULTS produced by a
deterministic sub-agent. Your only job is to interpret those results.

Hard rules:
- Answer in the SAME LANGUAGE as the user's request.
- Interpret ONLY the results provided. If the data does not show something,
  do not say it.
- Name ONLY species present in the provided species list.
- Never invent a relationship, a divergence time, a similarity score, a
  support value, a model name, or a scientific reference.
- Never restate a number that is not in the provided results. Refer to the
  values, do not recompute or round them.
- For a similarity result: explain what the groups mean and which species
  cluster together.
- For a phylogenetic result: explain the topology and what the support
  values say about how well resolved it is.
- Mention any important limitation, including the warnings you are given
  and the fact that providers are mocked when that is stated.
- Be concise: 2 to 4 sentences, no headings, no markdown, no code fences.
- Output ONLY the explanation text.
"""


async def explain(
    *,
    instruction: str,
    feature: str,
    species: list[str],
    results: dict,
    warnings: list[str] | None = None,
    providers_are_mocked: bool | None = None,
    llm=None,
) -> str | None:
    """Return a grounded interpretation, or ``None`` if none can be trusted.

    Parameters are keyword-only and form the complete whitelist — see
    ``ALLOWED_INPUT_KEYS``.

    ``llm`` is injectable for tests: pass a stub implementing
    ``ainvoke([SystemMessage, HumanMessage])`` to avoid any network call.
    """
    from langchain_core.messages import HumanMessage, SystemMessage

    if not results:
        return None

    if llm is None:
        try:
            from .framework.llm_client import get_llm
            llm = get_llm()
        except Exception as exc:  # noqa: BLE001
            _logger.warning("[Explainer] no LLM backend configured: %s", exc)
            return None

    payload = {
        "user_request": instruction,
        "feature": feature,
        "species": list(species),
        "results": results,
        "warnings": list(warnings or []),
        "providers_are_mocked": providers_are_mocked,
    }

    try:
        response = await llm.ainvoke(
            [
                SystemMessage(content=EXPLAINER_SYSTEM_PROMPT),
                HumanMessage(content=json.dumps(payload, indent=2, default=str)),
            ]
        )
    except Exception as exc:  # noqa: BLE001
        _logger.warning("[Explainer] LLM call failed (%s)", type(exc).__name__)
        return None

    text = getattr(response, "content", None)
    if not isinstance(text, str):
        _logger.info("[Explainer] non-text response of type %s", type(text).__name__)
        return None

    text = text.strip()
    if not text:
        _logger.info("[Explainer] empty response")
        return None

    if len(text) > _MAX_CHARS:
        _logger.info("[Explainer] response too long (%d chars)", len(text))
        return None

    problem = _grounding_violation(text, species, results)
    if problem:
        _logger.info("[Explainer] ungrounded response: %s", problem)
        return None

    return text


# ---------------------------------------------------------------------------
# Grounding guards — deterministic, run after the model answered
# ---------------------------------------------------------------------------

def _grounding_violation(
    text: str, species: list[str], results: dict
) -> str | None:
    """Return a reason string if ``text`` is not supported by the data.

    Two precise checks, chosen because they have no false positives:

    * every decimal number in the text must appear in the results (catches
      fabricated similarity scores, support values and confidences);
    * a binomial that shares a genus with an analysed species but is not one
      of them is a hallucinated taxon (catches "Homo erectus" when only
      "Homo sapiens" was analysed).

    Plain integers are allowed: "3 species", "two groups" are legitimate
    prose, not fabricated measurements.
    """
    blob = json.dumps(results, default=str)

    known_decimals = set(re.findall(r"\d+\.\d+", blob))
    for value in re.findall(r"\d+\.\d+", text):
        if value not in known_decimals and value.rstrip("0").rstrip(".") not in {
            d.rstrip("0").rstrip(".") for d in known_decimals
        }:
            return f"unknown numeric value {value!r}"

    allowed = {s.strip().lower() for s in species}
    allowed_genera = {s.split()[0] for s in allowed if s.split()}
    allowed_epithets = {p[1] for p in (s.split() for s in allowed) if len(p) > 1}
    for match in re.findall(r"\b([A-Z][a-z]{2,})\s+([a-z]{3,})\b", text):
        genus, second = match[0].lower(), match[1].lower()
        if f"{genus} {second}" in allowed:
            continue
        if genus not in allowed_genera:
            continue
        if second in allowed_epithets:
            # e.g. "Mus musculus" written after "Homo sapiens" — both known.
            continue
        if second in _PROSE_AFTER_GENUS:
            # "the Mus+Gallus node has ...", "the Homo branch is ..." — the
            # word after the genus is ordinary English, not an epithet.
            continue
        return f"species not in the analysed set: {genus} {second!r}"

    return None


# ---------------------------------------------------------------------------
# Deterministic fallback — no LLM involved
# ---------------------------------------------------------------------------

def fallback_explanation(feature: str, species: list[str], results: dict) -> str:
    """Minimal, fully deterministic sentence built from the worker output."""
    n = len(species)
    parts: list[str] = [f"Analysed {n} species."]

    similarity = results.get("similarity") or {}
    scores = similarity.get("similarity_scores") or []
    if scores:
        top = max(scores, key=lambda e: e.get("score", 0))
        parts.append(
            f"Closest pair: {top.get('species_a')} and {top.get('species_b')} "
            f"(similarity {top.get('score')})."
        )
    groups = similarity.get("species_groups") or []
    if groups:
        parts.append(f"{len(groups)} similarity group(s) found.")

    phylogeny = results.get("phylogeny") or {}
    if phylogeny:
        parts.append(f"Tree model: {phylogeny.get('model')}.")
        support = phylogeny.get("bootstrap_support") or {}
        if support:
            parts.append(f"{len(support)} supported internal node(s).")

    return " ".join(parts)
