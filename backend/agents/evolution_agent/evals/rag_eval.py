"""RAG Evaluation for the Evolution Agent — Sprint 4, Task 2.

The Evolution Agent's retrieval component is the Species Resolver
(``orchestrator/services/species_resolver.py``). It maps free-text species
names (common names, scientific names, aliases, typos, partial names) to
canonical scientific names that the downstream workers and external APIs
(GenBank, TimeTree) require.

This is the RAG "retrieval" step in the agent's pipeline:

    User query  →  Species Resolver (offline dict / Qdrant)
                        ↓
              Canonical scientific name
                        ↓
    Worker (MC / Phylo) uses the canonical name to fetch sequences

Why this is RAG
---------------
The resolver is a retrieval system: given a natural-language query it
searches a knowledge base (the offline catalogue, or a Qdrant vector
collection in Sprint 3+) and returns the most relevant document (the
canonical scientific name). The same four RAGAS metrics apply:

  1. Retrieval Relevance   — did it return the right species at all?
  2. Context Quality       — is the returned name unambiguous / correct?
  3. Answer Faithfulness   — is the canonical name actually in the catalogue?
                             (no hallucinated names)
  4. Answer Relevance      — does the canonical name answer what the user
                             asked? (e.g. "chimp" → "Pan troglodytes", not
                             a different primate)

All four are measured without an external LLM — scores are deterministic
and derived from the ground-truth dataset below.

Usage
-----
Run directly (no server required, no API key needed):

    python -m backend.agents.evolution_agent.evals.rag_eval

Writes ``rag_report.json`` and ``rag_report.md`` next to this file.
"""

from __future__ import annotations

import importlib.util as _ilu
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

# Import species_resolver directly by file path — this avoids triggering
# orchestrator/__init__.py which imports torch and langgraph, and would
# fail without a full GPU/ML environment.
_resolver_path = (
    Path(__file__).resolve().parents[1]
    / "orchestrator" / "services" / "species_resolver.py"
)
_spec = _ilu.spec_from_file_location("species_resolver", _resolver_path)
_mod  = _ilu.module_from_spec(_spec)   # type: ignore[arg-type]
sys.modules["species_resolver"] = _mod  # register before exec so dataclasses can find the module
_spec.loader.exec_module(_mod)          # type: ignore[union-attr]

SpeciesResolverService = _mod.SpeciesResolverService
_OfflineBackend        = _mod._OfflineBackend
_OFFLINE_CATALOGUE     = _mod._OFFLINE   # {lowercase_query: canonical}


# ---------------------------------------------------------------------------
# Ground-truth dataset
# ---------------------------------------------------------------------------
# Each case represents one "retrieval" call:
#   query            — raw input the user (or planner) might supply
#   expected         — canonical scientific name the resolver must return
#   expected_score   — minimum retrieval score we accept (1.0 = exact,
#                      0.6 = substring fallback; None = must fail)
#   category         — query type for grouping in the report
#   notes            — human-readable explanation of what is being tested

@dataclass
class RAGCase:
    id: str
    query: str
    expected: str | None          # None = must return None (unknown species)
    min_score: float | None       # None = not applicable (expected to fail)
    category: str
    notes: str = ""


CASES: list[RAGCase] = [
    # ── Exact scientific name ─────────────────────────────────────────────
    RAGCase(
        id="exact_homo_sapiens",
        query="Homo sapiens",
        expected="Homo sapiens",
        min_score=1.0,
        category="exact_scientific",
        notes="Title-cased scientific name — exact dict hit, score must be 1.0.",
    ),
    RAGCase(
        id="exact_pan_troglodytes",
        query="Pan troglodytes",
        expected="Pan troglodytes",
        min_score=1.0,
        category="exact_scientific",
    ),
    RAGCase(
        id="exact_mus_musculus",
        query="Mus musculus",
        expected="Mus musculus",
        min_score=1.0,
        category="exact_scientific",
    ),
    RAGCase(
        id="exact_gallus_gallus",
        query="Gallus gallus",
        expected="Gallus gallus",
        min_score=1.0,
        category="exact_scientific",
    ),
    RAGCase(
        id="exact_danio_rerio",
        query="Danio rerio",
        expected="Danio rerio",
        min_score=1.0,
        category="exact_scientific",
    ),
    # ── Lowercase scientific name ─────────────────────────────────────────
    RAGCase(
        id="lower_homo_sapiens",
        query="homo sapiens",
        expected="Homo sapiens",
        min_score=1.0,
        category="case_normalisation",
        notes="Lowercase scientific name must resolve to title-cased canonical.",
    ),
    RAGCase(
        id="lower_mus_musculus",
        query="mus musculus",
        expected="Mus musculus",
        min_score=1.0,
        category="case_normalisation",
    ),
    # ── Common names (primary aliases) ────────────────────────────────────
    RAGCase(
        id="common_human",
        query="human",
        expected="Homo sapiens",
        min_score=1.0,
        category="common_name",
        notes="Most common alias — must resolve exactly.",
    ),
    RAGCase(
        id="common_chimp",
        query="chimp",
        expected="Pan troglodytes",
        min_score=1.0,
        category="common_name",
    ),
    RAGCase(
        id="common_chimpanzee",
        query="chimpanzee",
        expected="Pan troglodytes",
        min_score=1.0,
        category="common_name",
    ),
    RAGCase(
        id="common_mouse",
        query="mouse",
        expected="Mus musculus",
        min_score=1.0,
        category="common_name",
    ),
    RAGCase(
        id="common_chicken",
        query="chicken",
        expected="Gallus gallus",
        min_score=1.0,
        category="common_name",
    ),
    RAGCase(
        id="common_zebrafish",
        query="zebrafish",
        expected="Danio rerio",
        min_score=1.0,
        category="common_name",
    ),
    # ── Secondary aliases ─────────────────────────────────────────────────
    RAGCase(
        id="alias_humans",
        query="humans",
        expected="Homo sapiens",
        min_score=1.0,
        category="alias",
        notes="Plural form — separate entry in catalogue.",
    ),
    RAGCase(
        id="alias_human_being",
        query="human being",
        expected="Homo sapiens",
        min_score=1.0,
        category="alias",
    ),
    RAGCase(
        id="alias_common_chimpanzee",
        query="common chimpanzee",
        expected="Pan troglodytes",
        min_score=1.0,
        category="alias",
    ),
    RAGCase(
        id="alias_house_mouse",
        query="house mouse",
        expected="Mus musculus",
        min_score=1.0,
        category="alias",
    ),
    RAGCase(
        id="alias_laboratory_mouse",
        query="laboratory mouse",
        expected="Mus musculus",
        min_score=1.0,
        category="alias",
    ),
    RAGCase(
        id="alias_red_junglefowl",
        query="red junglefowl",
        expected="Gallus gallus",
        min_score=1.0,
        category="alias",
    ),
    RAGCase(
        id="alias_zebra_fish",
        query="zebra fish",
        expected="Danio rerio",
        min_score=1.0,
        category="alias",
        notes="Two-word variant of zebrafish.",
    ),
    RAGCase(
        id="alias_zebra_danio",
        query="zebra danio",
        expected="Danio rerio",
        min_score=1.0,
        category="alias",
    ),
    # ── Substring / fuzzy fallback ────────────────────────────────────────
    RAGCase(
        id="fuzzy_the_common_chimpanzee",
        query="the common chimpanzee",
        expected="Pan troglodytes",
        min_score=0.6,
        category="fuzzy",
        notes=(
            "Substring fallback: 'common chimpanzee' is a substring of the "
            "query so the resolver should still find Pan troglodytes at score 0.6."
        ),
    ),
    RAGCase(
        id="fuzzy_laboratory_mouse_strain",
        query="laboratory mouse strain",
        expected="Mus musculus",
        min_score=0.6,
        category="fuzzy",
        notes="'laboratory mouse' appears inside the query string.",
    ),
    # ── Case-insensitive input ────────────────────────────────────────────
    RAGCase(
        id="mixed_case_HUMAN",
        query="HUMAN",
        expected="Homo sapiens",
        min_score=1.0,
        category="case_normalisation",
        notes="All-caps input normalised to lowercase before lookup.",
    ),
    RAGCase(
        id="mixed_case_ZebraFish",
        query="ZebraFish",
        expected="Danio rerio",
        min_score=1.0,
        category="case_normalisation",
    ),
    # ── Whitespace edge cases ─────────────────────────────────────────────
    RAGCase(
        id="whitespace_leading_trailing",
        query="  homo sapiens  ",
        expected="Homo sapiens",
        min_score=1.0,
        category="edge_case",
        notes="Leading/trailing whitespace is stripped by resolve().",
    ),
    RAGCase(
        id="empty_string",
        query="",
        expected=None,
        min_score=None,
        category="edge_case",
        notes="Empty string must return None — not crash.",
    ),
    RAGCase(
        id="whitespace_only",
        query="   ",
        expected=None,
        min_score=None,
        category="edge_case",
        notes="Whitespace-only string must return None — not crash.",
    ),
    # ── Unknown / out-of-catalogue species ────────────────────────────────
    RAGCase(
        id="unknown_dog",
        query="dog",
        expected=None,
        min_score=None,
        category="unknown",
        notes=(
            "Dog (Canis lupus familiaris) is not in the offline catalogue. "
            "Must return None cleanly — not hallucinate a species."
        ),
    ),
    RAGCase(
        id="unknown_cat",
        query="cat",
        expected=None,
        min_score=None,
        category="unknown",
    ),
    RAGCase(
        id="unknown_elephant",
        query="elephant",
        expected=None,
        min_score=None,
        category="unknown",
    ),
    RAGCase(
        id="unknown_scientific_gorilla",
        query="Gorilla gorilla",
        expected=None,
        min_score=None,
        category="unknown",
        notes=(
            "Gorilla is not in the catalogue even though it shares a genus "
            "with common primates. Must not be confused with Pan troglodytes."
        ),
    ),
    RAGCase(
        id="unknown_gibberish",
        query="xyzzy blorp",
        expected=None,
        min_score=None,
        category="unknown",
        notes="Complete gibberish — must return None without crashing.",
    ),
    # ── Batch resolution (resolve_all) ────────────────────────────────────
    # These test the batch API directly. The individual lookups are tested
    # above; here we check that the list contract is correct.
    RAGCase(
        id="batch_all_known",
        query="human|chimp|mouse",          # pipe-delimited — parsed by runner
        expected="Homo sapiens|Pan troglodytes|Mus musculus",
        min_score=1.0,
        category="batch",
        notes="All known species — unresolved list must be empty.",
    ),
    RAGCase(
        id="batch_mixed_known_unknown",
        query="human|dog",
        expected="Homo sapiens|<unresolved:dog>",
        min_score=None,
        category="batch",
        notes=(
            "Mixed batch: 'human' resolves, 'dog' does not. "
            "canonical=['Homo sapiens'], unresolved=['dog']."
        ),
    ),
    RAGCase(
        id="batch_all_unknown",
        query="dog|cat",
        expected="<unresolved:dog>|<unresolved:cat>",
        min_score=None,
        category="batch",
        notes="All unknown — canonical list must be empty.",
    ),
]


# ---------------------------------------------------------------------------
# RAGAS-style metrics (deterministic, no LLM)
# ---------------------------------------------------------------------------

@dataclass
class MetricResult:
    retrieval_relevance: float    # 1.0 = returned the right document, 0.0 = wrong/None
    context_quality:     float    # 1.0 = canonical name well-formed, 0.5 = partial
    answer_faithfulness: float    # 1.0 = name exists in catalogue, 0.0 = hallucinated
    answer_relevance:    float    # 1.0 = name matches the query intent


@dataclass
class CaseResult:
    case_id:   str
    query:     str
    expected:  str | None
    actual:    str | None
    score:     float | None
    passed:    bool
    metrics:   MetricResult
    detail:    str
    category:  str
    notes:     str = ""


# All valid canonical names — derived from the live catalogue so this eval
# never drifts from the resolver's actual knowledge base.
_CANONICAL_NAMES: set[str] = set(_OFFLINE_CATALOGUE.values())


def _is_well_formed(name: str) -> bool:
    """Genus + epithet, both capitalised correctly."""
    parts = name.split()
    if len(parts) != 2:
        return False
    genus, epithet = parts
    return genus[0].isupper() and epithet[0].islower()


def _score_single(case: RAGCase, actual_canonical: str | None, actual_score: float | None) -> CaseResult:
    """Compute the four RAGAS-style metrics for a single resolve() call."""

    # Expected-to-fail cases (unknown species, empty string)
    if case.expected is None:
        passed = actual_canonical is None
        rr = 1.0 if passed else 0.0   # retrieved nothing = correct for unknowns
        cq = 1.0 if passed else 0.0
        af = 1.0 if passed else 0.0   # not hallucinating a name = faithful
        ar = 1.0 if passed else 0.0
        detail = (
            "correctly returned None" if passed
            else f"should have returned None but got {actual_canonical!r}"
        )
        return CaseResult(
            case_id=case.id, query=case.query,
            expected=None, actual=actual_canonical, score=actual_score,
            passed=passed,
            metrics=MetricResult(rr, cq, af, ar),
            detail=detail, category=case.category, notes=case.notes,
        )

    # Expected-to-succeed cases
    correct = actual_canonical == case.expected

    # 1. Retrieval Relevance: did we get the right canonical name?
    rr = 1.0 if correct else 0.0

    # 2. Context Quality: is the returned name well-formed?
    if actual_canonical is None:
        cq = 0.0
    elif _is_well_formed(actual_canonical):
        cq = 1.0
    else:
        cq = 0.5   # returned something but malformed

    # 3. Answer Faithfulness: is the returned name actually in the catalogue?
    if actual_canonical is None:
        af = 0.0
    elif actual_canonical in _CANONICAL_NAMES:
        af = 1.0
    else:
        af = 0.0   # hallucinated name not in catalogue

    # 4. Answer Relevance: does the answer match what was asked?
    #    Scored as 1.0 if correct, 0.5 if we got a real species but the
    #    wrong one (wrong but faithful), 0.0 if None.
    if correct:
        ar = 1.0
    elif actual_canonical is not None and actual_canonical in _CANONICAL_NAMES:
        ar = 0.5
    else:
        ar = 0.0

    score_ok = (
        actual_score is not None
        and case.min_score is not None
        and actual_score >= case.min_score
    )

    passed = correct and score_ok

    detail_parts = [f"expected={case.expected!r} actual={actual_canonical!r}"]
    if actual_score is not None:
        detail_parts.append(
            f"score={actual_score:.2f} (min={case.min_score})"
            + (" ✓" if score_ok else " ✗")
        )
    if not correct:
        detail_parts.append("WRONG CANONICAL NAME")

    return CaseResult(
        case_id=case.id, query=case.query,
        expected=case.expected, actual=actual_canonical, score=actual_score,
        passed=passed,
        metrics=MetricResult(rr, cq, af, ar),
        detail="; ".join(detail_parts),
        category=case.category, notes=case.notes,
    )


def _score_batch(case: RAGCase, svc: SpeciesResolverService) -> CaseResult:
    """Score a batch (pipe-delimited) case using resolve_all()."""
    queries = case.query.split("|")
    canonical_list, unresolved_list = svc.resolve_all(queries)

    # Decode expected string into what we want
    expected_parts = (case.expected or "").split("|")
    expected_canonical = [p for p in expected_parts if not p.startswith("<unresolved:")]
    expected_unresolved = [
        p[len("<unresolved:"):].rstrip(">")
        for p in expected_parts if p.startswith("<unresolved:")
    ]

    canonical_ok = sorted(canonical_list) == sorted(expected_canonical)
    unresolved_ok = sorted(unresolved_list) == sorted(expected_unresolved)
    passed = canonical_ok and unresolved_ok

    rr = 1.0 if canonical_ok else 0.0
    cq = 1.0 if all(_is_well_formed(n) for n in canonical_list) else 0.5
    af = 1.0 if all(n in _CANONICAL_NAMES for n in canonical_list) else 0.0
    ar = 1.0 if passed else (0.5 if canonical_ok else 0.0)

    detail = (
        f"canonical={canonical_list} (expected={expected_canonical}); "
        f"unresolved={unresolved_list} (expected={expected_unresolved})"
    )
    return CaseResult(
        case_id=case.id, query=case.query,
        expected=case.expected, actual="|".join(canonical_list),
        score=None, passed=passed,
        metrics=MetricResult(rr, cq, af, ar),
        detail=detail, category=case.category, notes=case.notes,
    )


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run() -> list[CaseResult]:
    svc = SpeciesResolverService(backend=_OfflineBackend())
    results: list[CaseResult] = []

    for case in CASES:
        if case.category == "batch":
            results.append(_score_batch(case, svc))
            continue

        resolved = svc.resolve(case.query)
        actual_canonical = resolved.canonical if resolved else None
        actual_score = resolved.score if resolved else None
        results.append(_score_single(case, actual_canonical, actual_score))

    return results


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _mean(values: list[float]) -> float:
    return round(sum(values) / len(values), 3) if values else 0.0


def write_reports(results: list[CaseResult], out_dir: Path) -> None:
    json_path = out_dir / "rag_report.json"
    md_path   = out_dir / "rag_report.md"

    json_path.write_text(
        json.dumps([asdict(r) for r in results], indent=2),
        encoding="utf-8",
    )

    total  = len(results)
    passed = sum(1 for r in results if r.passed)

    rr_scores  = [r.metrics.retrieval_relevance for r in results]
    cq_scores  = [r.metrics.context_quality     for r in results]
    af_scores  = [r.metrics.answer_faithfulness for r in results]
    ar_scores  = [r.metrics.answer_relevance    for r in results]

    # Group pass rate by category
    categories: dict[str, list[CaseResult]] = {}
    for r in results:
        categories.setdefault(r.category, []).append(r)

    lines: list[str] = [
        "# Evolution Agent — RAG Evaluation Report (Species Resolver)",
        "",
        "## Overview",
        "",
        "The Species Resolver is the retrieval component of the Evolution Agent.",
        "It maps free-text species names (common names, aliases, scientific names)",
        "to canonical scientific names that the downstream workers require.",
        "",
        "This evaluation uses RAGAS-style metrics measured deterministically",
        "(no LLM required). The offline backend is evaluated here; the Qdrant",
        "vector backend (Sprint 3+) will be evaluated when the cluster is live.",
        "",
        "## RAGAS Metrics",
        "",
        "| Metric | Definition | Score |",
        "|---|---|---|",
        f"| Retrieval Relevance   | Returned the correct canonical name | {_mean(rr_scores):.3f} |",
        f"| Context Quality       | Returned name is well-formed (Genus epithet) | {_mean(cq_scores):.3f} |",
        f"| Answer Faithfulness   | Returned name exists in the knowledge base | {_mean(af_scores):.3f} |",
        f"| Answer Relevance      | Returned name answers the user's query intent | {_mean(ar_scores):.3f} |",
        "",
        f"**Overall: {passed}/{total} cases passed**",
        "",
        "## Pass Rate by Category",
        "",
        "| Category | Passed | Total |",
        "|---|---|---|",
    ]

    for cat, cat_results in sorted(categories.items()):
        cat_passed = sum(1 for r in cat_results if r.passed)
        lines.append(f"| {cat} | {cat_passed} | {len(cat_results)} |")

    # Failures
    failures = [r for r in results if not r.passed]
    lines += [
        "",
        "## Failures and Weaknesses",
        "",
    ]
    if failures:
        for r in failures:
            lines.append(f"### `{r.case_id}` ({r.category})")
            lines.append(f"- Query: `{r.query}`")
            lines.append(f"- Expected: `{r.expected}`")
            lines.append(f"- Actual:   `{r.actual}` (score={r.score})")
            lines.append(f"- Detail: {r.detail}")
            if r.notes:
                lines.append(f"- Notes: {r.notes}")
            lines.append(f"- Metrics: RR={r.metrics.retrieval_relevance} CQ={r.metrics.context_quality} AF={r.metrics.answer_faithfulness} AR={r.metrics.answer_relevance}")
            lines.append("")
    else:
        lines.append("No failures — all cases passed.")
        lines.append("")

    # Known limitations
    lines += [
        "## Known Limitations",
        "",
        "1. **Catalogue coverage**: only 5 species are in the offline catalogue.",
        "   Any species outside this set fails at the resolver, not the worker.",
        "   Sprint 3 will replace the offline dict with a Qdrant vector search",
        "   over ESMC embeddings, extending coverage to any species with a",
        "   GenBank entry.",
        "",
        "2. **Substring fallback ambiguity**: the fuzzy fallback uses substring",
        "   matching (`key in query or query in key`). A query like `'pan'`",
        "   would match `'pan troglodytes'` with score 0.6, which is correct,",
        "   but `'mouse'` inside `'house mouse'` also matches, producing the",
        "   right answer for the wrong reason. The Qdrant backend will resolve",
        "   this with semantic similarity instead.",
        "",
        "3. **No typo tolerance**: `'chimpanzea'` (one character off) returns",
        "   None because neither exact nor substring matching handles typos.",
        "   The Qdrant backend with cosine similarity will handle this.",
        "",
        "4. **Qdrant backend not evaluated**: `_QdrantBackend` requires a live",
        "   cluster and is excluded from this run. A separate evaluation run",
        "   with `QDRANT_URL` and `QDRANT_API_KEY` set will exercise it.",
        "",
        "## Per-Case Detail",
        "",
    ]

    for r in results:
        status = "PASS" if r.passed else "FAIL"
        lines.append(f"### `{r.case_id}` — {status}")
        lines.append(f"- Query: `{r.query}`")
        lines.append(f"- Expected: `{r.expected}`  Actual: `{r.actual}`  Score: {r.score}")
        lines.append(f"- Detail: {r.detail}")
        if r.notes:
            lines.append(f"- Notes: {r.notes}")
        lines.append(
            f"- RR={r.metrics.retrieval_relevance:.1f}  "
            f"CQ={r.metrics.context_quality:.1f}  "
            f"AF={r.metrics.answer_faithfulness:.1f}  "
            f"AR={r.metrics.answer_relevance:.1f}"
        )
        lines.append("")

    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[rag_eval] wrote {json_path}")
    print(f"[rag_eval] wrote {md_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    results = run()

    passed = sum(1 for r in results if r.passed)
    total  = len(results)
    print(f"\n[rag_eval] {passed}/{total} cases passed")

    failures = [r for r in results if not r.passed]
    if failures:
        print("\nFailed cases:")
        for r in failures:
            print(f"  {r.case_id}: {r.detail}")

    write_reports(results, Path(__file__).resolve().parent)
