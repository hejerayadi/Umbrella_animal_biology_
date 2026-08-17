"""
Gene Annotation — §3.10 "Test independently" runner (REAL execution).

This is not a mock-heavy unit test. It calls the real get_gene_annotation()
against live NCBI Gene eutils, and calls the real LLM strategy resolver
(resolve_gene_annotation_strategy) against the real NVIDIA endpoint when
NVIDIA_API_KEY is set — exactly the code path that runs in production.
Nothing about the LLM's *decision* is mocked or scripted.

There is exactly one thing this script deliberately forces, because the
design spec's own §3.10 case for it requires simulating the LLM being
unavailable: case 3 patches get_llm_client() to raise, the same failure
mode you'd see with a missing/invalid API key or a saturated free-tier
endpoint. Every other case makes real calls end to end.

    1. General question       -> real-description genes rank ahead of
                                  uncharacterized ones (checked as a
                                  general ordering invariant against
                                  whatever NCBI actually returns).
    2. Trait-specific question -> the real LLM (or, if no key, the
                                  deterministic keyword fallback) derives
                                  a search keyword from the question.
    3. LLM unavailable         -> forced failure of get_llm_client();
                                  deterministic fallback still returns a
                                  valid, UNRANKED table (§3.9: "no ranking,
                                  first 50 as-is").
    4. Empty NCBI description  -> grounding check: every function value in
                                  the real result matches NCBI's own raw
                                  esummary response, verbatim, including
                                  genes where that field is empty.

Requires network access to eutils.ncbi.nlm.nih.gov (unauthenticated).
NVIDIA_API_KEY is optional: if set in .env, cases 1/2/4 exercise the real
LLM; if absent, they exercise the real deterministic fallback instead —
either way this reflects actual production behavior, not a simulation.

Run:
    python -m genome_agent.scripts.run_gene_annotation_independent_tests

or, from inside the container:
    python scripts/run_gene_annotation_independent_tests.py

Options:
    --species NAME   Species to test against (default: house mouse — a
                      well-annotated model organism, so a broad search
                      reliably returns a mix of informative and
                      "uncharacterized LOC..." genes).
"""
from __future__ import annotations

import argparse
import asyncio
import contextlib
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Callable
from unittest.mock import patch

# Root stays at WARNING so third-party libraries (httpx, langchain internals,
# etc.) don't spam the console. The two loggers that actually tell us whether
# the *real* LLM call succeeded or silently fell back are captured separately
# below at INFO, regardless of this root setting — see _capture_llm_logs().
logging.getLogger().setLevel(logging.WARNING)

_RESOLVER_LOGGER_NAME = "genome_agent.workflows.gene_annotation_resolver"
_LLM_LOGGER_NAME = "genome_agent.workflows.llm"


class _RecordCollector(logging.Handler):
    """Collects log records in memory instead of printing them, so a case
    can inspect exactly what the resolver/llm modules logged internally —
    including INFO-level messages that the root WARNING threshold would
    otherwise swallow silently."""

    def __init__(self) -> None:
        super().__init__()
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


@contextlib.contextmanager
def _capture_llm_logs():
    """Temporarily raise the resolver/llm loggers to INFO and capture their
    output, so real LLM failures (bad key, wrong model name, network block,
    timeout, etc.) are visible in the case's own report instead of being
    silently absorbed by the fallback and hidden behind a 'PASS'."""
    collector = _RecordCollector()
    collector.setLevel(logging.INFO)
    loggers = [logging.getLogger(_RESOLVER_LOGGER_NAME), logging.getLogger(_LLM_LOGGER_NAME)]
    saved = [(lg, lg.level) for lg in loggers]
    for lg in loggers:
        lg.addHandler(collector)
        lg.setLevel(logging.INFO)
    try:
        yield collector
    finally:
        for lg, level in saved:
            lg.removeHandler(collector)
            lg.setLevel(level)


def _summarize_llm_usage(collector: "_RecordCollector") -> str:
    """Turn captured log records into one clear line: did this case's call
    actually use the live LLM, or silently fall back — and if it fell back,
    why?"""
    for record in collector.records:
        message = record.getMessage()
        if record.name == _RESOLVER_LOGGER_NAME and "LLM client unavailable" in message:
            return f"FALLBACK (client could not be built: {message.split(': ', 1)[-1]})"
        if record.name == _RESOLVER_LOGGER_NAME and "resolver unavailable" in message:
            return f"FALLBACK (invoke failed: {message.split('(', 1)[-1].rsplit(')', 1)[0]})"
    return "REAL LLM (no fallback logged)"

_WIDTH = 72

try:
    from ..subagents.gene_annotation import (
        _is_informative_description,
        fetch_gene_summaries,
        get_gene_annotation,
        search_genes,
    )
    from ..subagents._ncbi_client import ncbi_get
    from ..subagents.species_resolver import resolve_species
except ImportError:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
    from genome_agent.subagents.gene_annotation import (
        _is_informative_description,
        fetch_gene_summaries,
        get_gene_annotation,
        search_genes,
    )
    from genome_agent.subagents._ncbi_client import ncbi_get
    from genome_agent.subagents.species_resolver import resolve_species

_MOD = "genome_agent.subagents.gene_annotation"
_RESOLVER_MOD = "genome_agent.workflows.gene_annotation_resolver"


@dataclass
class CaseResult:
    slug: str
    title: str
    passed: bool
    detail_lines: list[str] = field(default_factory=list)
    error: str | None = None


def _print_case(result: CaseResult) -> None:
    status = "PASS ✅" if result.passed else "FAIL ❌"
    print("\n" + "=" * _WIDTH)
    print(f"CASE: {result.title}")
    print("=" * _WIDTH)
    for line in result.detail_lines:
        print(f"  {line}")
    if result.error:
        print(f"  ERROR: {result.error}")
    print(f"Result: {status}")
    print("=" * _WIDTH)


async def case_general_question_ranks_real_descriptions_first(assembly_id: str) -> CaseResult:
    """§3.10 case 1 (real): general question -> real-description genes rank
    ahead of uncharacterized ones. Checked as a general invariant over
    whatever NCBI actually returns right now, not fixed expected genes."""
    slug = "general-question-ranking"
    title = "General question -> real-description genes rank ahead of uncharacterized ones (LIVE NCBI + LLM)"
    detail: list[str] = []

    try:
        with _capture_llm_logs() as collector:
            result = await get_gene_annotation(
                assembly_id,
                user_question="Show me gene annotations for this species",
            )
        detail.append(f"LLM path actually taken: {_summarize_llm_usage(collector)}")
        gene_table = result["gene_table"]
        detail.append(f"NCBI returned {len(gene_table)} genes for assembly {assembly_id!r}")

        informative_flags = [
            _is_informative_description(row.get("gene_name", ""), row.get("function", ""))
            for row in gene_table
        ]
        n_informative = sum(informative_flags)
        n_uncharacterized = len(informative_flags) - n_informative
        detail.append(f"informative-description genes: {n_informative}, uncharacterized: {n_uncharacterized}")
        detail.append(
            "sample (first 5): "
            + ", ".join(f"{row['gene_name']}({'real' if flag else 'uncharacterized'})"
                         for row, flag in list(zip(gene_table, informative_flags))[:5])
        )

        if n_informative == 0 or n_uncharacterized == 0:
            detail.append(
                "NOTE: live NCBI data for this assembly didn't contain both a real-description "
                "and an uncharacterized gene in this batch, so the ordering invariant is vacuously "
                "satisfied — rerun with a different --species for a stronger check if you want one."
            )
            return CaseResult(slug, title, True, detail)

        # The core invariant: no uncharacterized gene may appear before any
        # informative-description gene, wherever the boundary falls.
        last_informative_index = max(i for i, flag in enumerate(informative_flags) if flag)
        first_uncharacterized_index = min(i for i, flag in enumerate(informative_flags) if not flag)
        ranked_correctly = last_informative_index < first_uncharacterized_index
        detail.append(f"last informative-gene index: {last_informative_index}")
        detail.append(f"first uncharacterized-gene index: {first_uncharacterized_index}")
        detail.append(f"all informative genes precede all uncharacterized genes: {ranked_correctly}")

        return CaseResult(slug, title, ranked_correctly, detail)
    except Exception as exc:  # pragma: no cover
        return CaseResult(slug, title, False, detail, error=repr(exc))


async def case_trait_specific_question_uses_keyword(assembly_id: str) -> CaseResult:
    """§3.10 case 2 (real): trait-specific question -> search_genes was
    called with a keyword derived from the question, by the real LLM (or
    the real deterministic fallback if no key is set)."""
    slug = "trait-specific-keyword"
    title = "Trait-specific question -> a keyword is derived from the question (LIVE LLM/fallback + NCBI)"
    detail: list[str] = []

    captured_terms: list[str] = []
    real_ncbi_get = ncbi_get

    def _spy_ncbi_get(params: dict):
        if params.get("db") == "gene" and "term" in params:
            captured_terms.append(params["term"])
        return real_ncbi_get(params)

    try:
        with patch(f"{_MOD}.ncbi_get", side_effect=_spy_ncbi_get), _capture_llm_logs() as collector:
            await get_gene_annotation(
                assembly_id,
                user_question="Which genes affect fur color in this species?",
            )

        detail.append(f"LLM path actually taken: {_summarize_llm_usage(collector)}")
        term = captured_terms[0] if captured_terms else ""
        detail.append(f"esearch term actually sent to NCBI: {term!r}")

        keyword_present = "[Gene Name]" in term
        detail.append(f"a derived keyword was included in the search term: {keyword_present}")

        return CaseResult(slug, title, keyword_present, detail)
    except Exception as exc:  # pragma: no cover
        return CaseResult(slug, title, False, detail, error=repr(exc))


async def case_llm_unavailable_falls_back_unranked(assembly_id: str) -> CaseResult:
    """§3.10 case 3: LLM unavailable (forced) -> deterministic broad-fetch
    fallback still returns a valid, UNRANKED table (§3.9: 'no ranking,
    first 50 as-is'). This is the one case that must force unavailability
    regardless of whether a real key is configured — that's what the case
    is testing. NCBI itself is still hit live."""
    slug = "llm-unavailable-fallback"
    title = "LLM unavailable (forced) -> deterministic fallback still returns a valid, unranked table (LIVE NCBI)"
    detail: list[str] = [
        "NOTE: this case intentionally forces get_llm_client() to raise — the "
        "'LLM client unavailable: Simulated: NVIDIA_API_KEY unavailable' warning "
        "you may see logged for this case is expected and is what's being tested here, "
        "not a real failure.",
    ]

    try:
        with patch(f"{_RESOLVER_MOD}.get_llm_client") as mock_llm:
            mock_llm.side_effect = EnvironmentError("Simulated: NVIDIA_API_KEY unavailable")

            result = await get_gene_annotation(
                assembly_id,
                user_question="Show me gene annotations for this species",
            )

        gene_table = result["gene_table"]
        detail.append(f"NCBI returned {len(gene_table)} genes (fallback path, LLM forced unavailable)")
        valid_table = len(gene_table) > 0
        detail.append(f"fallback returned a non-empty, valid table: {valid_table}")

        # Confirm it's actually unranked: refetch the same IDs directly and
        # compare order against what get_gene_annotation returned.
        gene_ids = await search_genes(assembly_id, keyword=None, max_results=50)
        raw_table, _ = await fetch_gene_summaries(gene_ids)
        raw_order = [row["gene_name"] for row in raw_table]
        returned_order = [row["gene_name"] for row in gene_table]
        unranked = returned_order == raw_order
        detail.append(f"returned order matches raw NCBI order (unranked, as §3.9 requires): {unranked}")

        return CaseResult(slug, title, valid_table and unranked, detail)
    except Exception as exc:  # pragma: no cover
        return CaseResult(slug, title, False, detail, error=repr(exc))


async def case_empty_description_never_invented(assembly_id: str) -> CaseResult:
    """§3.10 case 4 (real): for every gene NCBI returns, the function field
    matches NCBI's own raw response verbatim — including genes with a
    genuinely empty description. Nothing is invented from general biology
    knowledge to "fill in" a blank field."""
    slug = "empty-description-not-invented"
    title = 'Every gene\'s function matches NCBI verbatim; empty descriptions stay "" (LIVE NCBI)'
    detail: list[str] = []

    try:
        with _capture_llm_logs() as collector:
            result = await get_gene_annotation(
                assembly_id,
                user_question="Show me gene annotations for this species",
            )
        detail.append(f"LLM path actually taken: {_summarize_llm_usage(collector)}")
        gene_table = result["gene_table"]

        # Independently re-fetch the same genes' raw NCBI records ourselves,
        # bypassing the agent entirely, to have an untouched ground truth
        # to compare against.
        gene_ids = await search_genes(assembly_id, keyword=None, max_results=50)
        raw_table, _ = await fetch_gene_summaries(gene_ids)
        raw_by_name = {row["gene_name"]: row["function"] for row in raw_table}

        mismatches = []
        empty_count = 0
        for row in gene_table:
            raw_function = raw_by_name.get(row["gene_name"])
            if raw_function == "":
                empty_count += 1
            if raw_function is not None and row["function"] != raw_function:
                mismatches.append((row["gene_name"], row["function"], raw_function))

        detail.append(f"genes checked: {len(gene_table)}, genes with empty NCBI description: {empty_count}")
        if mismatches:
            for name, got, raw in mismatches[:5]:
                detail.append(f"  MISMATCH {name}: agent returned {got!r}, NCBI raw was {raw!r}")
        no_invention = len(mismatches) == 0
        detail.append(f"no invented/altered function text found: {no_invention}")

        return CaseResult(slug, title, no_invention, detail)
    except Exception as exc:  # pragma: no cover
        return CaseResult(slug, title, False, detail, error=repr(exc))


async def run_all(species_name: str) -> int:
    llm_mode = (
        "LIVE (NVIDIA_API_KEY set)"
        if os.getenv("NVIDIA_API_KEY")
        else "FALLBACK (no NVIDIA_API_KEY — real deterministic fallback logic)"
    )
    print("\n" + "#" * _WIDTH)
    print("# Gene Annotation Agent — §3.10 'Test independently' execution (REAL)")
    print(f"# LLM mode: {llm_mode}")
    print(f"# Species: {species_name!r} (live NCBI eutils)")
    print("#" * _WIDTH)

    species = await resolve_species(species_name)
    assembly_id = species.get("assembly_id")
    if not assembly_id:
        print(f"\nCould not resolve {species_name!r} to an assembly_id — aborting.")
        return 1
    print(f"\nResolved {species_name!r} -> assembly_id={assembly_id!r} "
          f"({species.get('scientific_name')}, confidence={species.get('confidence')})")

    cases: list[Callable[[str], Any]] = [
        case_general_question_ranks_real_descriptions_first,
        case_trait_specific_question_uses_keyword,
        case_llm_unavailable_falls_back_unranked,
        case_empty_description_never_invented,
    ]

    results = []
    for case in cases:
        results.append(await case(assembly_id))

    for result in results:
        _print_case(result)

    passed = sum(1 for r in results if r.passed)
    total = len(results)

    print("\n" + "#" * _WIDTH)
    if os.getenv("NVIDIA_API_KEY"):
        print("# LLM usage by case (cases 1/2/4 are supposed to hit the real LLM;")
        print("# case 3 intentionally forces it off — see that case's own note):")
        for r in results:
            for line in r.detail_lines:
                if line.startswith("LLM path actually taken:"):
                    print(f"#   [{r.slug}] {line}")
        print("#" * _WIDTH)
    print(f"# {passed}/{total} case(s) passed.")
    if passed < total:
        print("# FAILING CASES:")
        for r in results:
            if not r.passed:
                print(f"#   - {r.title}")
    print("#" * _WIDTH)

    return 0 if passed == total else 1


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--species",
        default="house mouse",
        help="Species to test against (default: house mouse)",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()
    sys.exit(asyncio.run(run_all(args.species)))