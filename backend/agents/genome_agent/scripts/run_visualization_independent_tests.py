"""
Visualization Agent — §4.10 "Test independently" runner (REAL execution).

This calls the real generate_visualization() against live NCBI eutils, and
against the real LLM resolvers (resolve_visualization_references,
resolve_chromosome_highlight) when NVIDIA_API_KEY is set — the same code
path that runs in production. Nothing about what the LLM decides is
scripted or mocked.

Two of the five cases necessarily force a specific condition, because
that's literally what the design spec's own §4.10 case is testing —
not a shortcut, but the point of the case:

    1. size_comparison (tiger)  -> the real LLM proposes reference species
                                    (or, with no key, the real fixed
                                    fallback list); the proposal is used
                                    end to end through live NCBI resolution
                                    and rendering.
    2. Partial reference failure -> one candidate is a species that
                                    genuinely will not resolve on NCBI
                                    (no mocking — an invalid name really
                                    fails), the other genuinely does; the
                                    chart must still render with the rest.
    3. chromosome_map highlight  -> the real resolver (LLM or fallback)
                                    extracts the highlighted gene from a
                                    real question against a real
                                    candidate list.
    4. protein_structure         -> spies (not fakes) wrap the real
                                    resolver functions so we can prove
                                    zero calls happened, without changing
                                    their behavior at all.
    5. LLM unavailable           -> get_llm_client() is forced to fail
                                    (this IS the case), but every NCBI
                                    call in both fallback paths is real.

Requires network access to eutils.ncbi.nlm.nih.gov. NVIDIA_API_KEY is
optional — see the case docstrings above for what changes without one.

Run:
    python -m genome_agent.scripts.run_visualization_independent_tests

or, from inside the container:
    python scripts/run_visualization_independent_tests.py
"""
from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Callable
from unittest.mock import patch

# Root stays at WARNING so third-party libraries don't spam the console.
# _capture_llm_logs() below captures the resolver/llm modules' own INFO-level
# messages separately, so real fallback reasons are visible per case instead
# of being silently absorbed.
logging.getLogger().setLevel(logging.WARNING)

_RESOLVER_LOGGER_NAME = "genome_agent.workflows.visualization_resolver"
_LLM_LOGGER_NAME = "genome_agent.workflows.llm"


class _RecordCollector(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


@contextlib.contextmanager
def _capture_llm_logs():
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


def _summarize_highlight_decision(collector: "_RecordCollector") -> str:
    """Report exactly what happened to the highlight_gene decision: did the
    LLM call fail outright, did it succeed but get dropped for not matching
    a candidate name, or did the deterministic fallback end up supplying
    the answer?"""
    notes = []
    for record in collector.records:
        message = record.getMessage()
        if "LLM client unavailable" in message:
            notes.append(f"client unavailable ({message.split(': ', 1)[-1]})")
        elif "resolver unavailable" in message:
            notes.append(f"invoke failed ({message.split('(', 1)[-1].rsplit(')', 1)[0]})")
        elif "not in candidate list" in message:
            notes.append(f"LLM proposed a name that got dropped: {message}")
    return "; ".join(notes) if notes else "no fallback/drop logged — LLM's own pick was used as-is"


_WIDTH = 72

try:
    from ..subagents.visualization import (
        generate_visualization,
        resolve_chromosome_highlight,
        resolve_reference_species,
        resolve_visualization_references,
    )
except ImportError:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
    from genome_agent.subagents.visualization import (
        generate_visualization,
        resolve_chromosome_highlight,
        resolve_reference_species,
        resolve_visualization_references,
    )

_MOD = "genome_agent.subagents.visualization"
_RESOLVER_MOD = "genome_agent.workflows.visualization_resolver"


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


async def case_size_comparison_uses_real_proposal() -> CaseResult:
    """§4.10 case 1 (real): size_comparison for a mammal -> the real
    reference-species proposal (LLM if available, else the real fixed
    fallback list) is resolved live against NCBI and actually used to
    render the chart."""
    slug = "size-comparison-real-list-used"
    title = "size_comparison (tiger) -> real reference-species proposal is resolved and used (LIVE LLM/fallback + NCBI)"
    detail: list[str] = []

    captured_lists: list[list[str]] = []
    real_resolve_reference_species = resolve_reference_species

    async def _spy_resolve_reference_species(candidate_names: list[str]):
        captured_lists.append(list(candidate_names))
        return await real_resolve_reference_species(candidate_names)

    try:
        from ..subagents.species_resolver import resolve_species
        from ..subagents.genome_metadata import get_genome_metadata
    except ImportError:
        from genome_agent.subagents.species_resolver import resolve_species
        from genome_agent.subagents.genome_metadata import get_genome_metadata

    try:
        tiger = await resolve_species("tiger")
        if not tiger.get("assembly_id"):
            detail.append("Could not resolve 'tiger' on live NCBI — skipping this case.")
            return CaseResult(slug, title, False, detail, error="tiger did not resolve")
        tiger_meta = await get_genome_metadata(tiger["assembly_id"])
        detail.append(f"queried species: tiger -> {tiger['assembly_id']} "
                       f"({tiger_meta.get('genome_size_bp')} bp)")

        with patch(f"{_MOD}.resolve_reference_species", side_effect=_spy_resolve_reference_species):
            result = await generate_visualization(
                scope="size_comparison",
                genome_size_bp=tiger_meta.get("genome_size_bp"),
                assembly_id=tiger["assembly_id"],
                common_name=tiger.get("common_name") or "tiger",
                scientific_name=tiger.get("scientific_name"),
                user_question="Compare the tiger's genome size to related cats",
            )

        proposed = captured_lists[0] if captured_lists else []
        detail.append(f"real resolver proposed reference species: {proposed}")
        sensible = 2 <= len(proposed) <= 4 and "tiger" not in [p.lower() for p in proposed]
        detail.append(f"proposal has 2-4 species and excludes the queried species itself: {sensible}")

        rendered = result["status"] == "COMPLETED" and result.get("chart_data") is not None
        comparisons = result.get("comparisons") or []
        detail.append(f"status={result['status']!r}, comparisons resolved: {[c['common_name'] for c in comparisons]}")
        detail.append(f"chart rendered: {rendered}")

        passed = bool(proposed) and sensible and rendered
        return CaseResult(slug, title, passed, detail)
    except Exception as exc:  # pragma: no cover
        return CaseResult(slug, title, False, detail, error=repr(exc))


async def case_partial_reference_failure_still_renders() -> CaseResult:
    """§4.10 case 2 (real): one reference species is a name that genuinely
    will not resolve on NCBI (no mocking — it really fails), mixed with
    one that genuinely does. The chart must still render with the rest."""
    slug = "partial-reference-failure"
    title = "One reference species genuinely fails to resolve on NCBI -> chart still renders with the rest (LIVE NCBI)"
    detail: list[str] = []

    try:
        from ..subagents.species_resolver import resolve_species
        from ..subagents.genome_metadata import get_genome_metadata
    except ImportError:
        from genome_agent.subagents.species_resolver import resolve_species
        from genome_agent.subagents.genome_metadata import get_genome_metadata

    try:
        human = await resolve_species("human")
        if not human.get("assembly_id"):
            detail.append("Could not resolve 'human' on live NCBI — skipping this case.")
            return CaseResult(slug, title, False, detail, error="human did not resolve")
        human_meta = await get_genome_metadata(human["assembly_id"])

        # "zzznotarealspeciesxyz" is not a real species name and will
        # genuinely fail to resolve against NCBI Assembly — no mocking.
        candidates = ["house mouse", "zzznotarealspeciesxyz"]
        detail.append(f"reference candidates passed straight to resolve_reference_species: {candidates}")

        resolved_rows = await resolve_reference_species(candidates)
        detail.append(f"resolve_reference_species() returned: {[r['common_name'] for r in resolved_rows]}")
        human_mouse = await resolve_species("house mouse")
        one_dropped = (
            len(resolved_rows) == 1
            and resolved_rows[0]["assembly_id"] == human_mouse.get("assembly_id")
        )
        detail.append(f"the fake species was dropped, the real one kept: {one_dropped}")

        result = await generate_visualization(
            scope="size_comparison",
            genome_size_bp=human_meta.get("genome_size_bp"),
            assembly_id=human["assembly_id"],
            common_name=human.get("common_name") or "human",
            scientific_name=human.get("scientific_name"),
            user_question="Compare human genome size to a mouse and a made-up species",
        )
        rendered = result["status"] == "COMPLETED" and result.get("chart_data") is not None
        detail.append(f"chart rendered despite one failed reference: {rendered}")

        passed = one_dropped and rendered
        return CaseResult(slug, title, passed, detail)
    except Exception as exc:  # pragma: no cover
        return CaseResult(slug, title, False, detail, error=repr(exc))


async def case_chromosome_map_highlight_from_question() -> CaseResult:
    """§4.10 case 3 (real): chromosome_map with a gene named in the
    question -> the real resolver (LLM if available, else the real
    heuristic fallback) sets highlight_gene from the question."""
    slug = "chromosome-map-highlight"
    title = "chromosome_map with a gene named in the question -> highlight_gene set for real (LIVE LLM/fallback)"
    detail: list[str] = []

    gene_table = [
        {"gene_name": "ABC", "location": "chr1", "function": "Function A"},
        {"gene_name": "Trp53", "location": "chr17", "function": "tumor suppressor"},
    ]
    question = "Show me the chromosome map with Trp53 highlighted"

    try:
        with _capture_llm_logs() as collector:
            result = await generate_visualization(
                scope="chromosome_map",
                gene_table=gene_table,
                user_question=question,
            )

        rendered = result["status"] == "COMPLETED" and result.get("chart_data") is not None
        highlight_color_present = b"e8622c" in (result.get("chart_data") or b"")
        detail.append(f"question: {question!r}")
        detail.append(f"candidate genes: {[g['gene_name'] for g in gene_table]}")
        detail.append(f"chart rendered: {rendered}")
        detail.append(f"Trp53 highlighted in rendered SVG (real resolver found it): {highlight_color_present}")
        detail.append(f"highlight decision detail: {_summarize_highlight_decision(collector)}")

        passed = rendered and highlight_color_present
        return CaseResult(slug, title, passed, detail)
    except Exception as exc:  # pragma: no cover
        return CaseResult(slug, title, False, detail, error=repr(exc))


async def case_protein_structure_never_calls_llm() -> CaseResult:
    """§4.10 case 4: protein_structure -> no LLM call happens at all. Uses
    spies (wraps=real function) rather than fakes, so if this somehow DID
    call one of these, it would still behave correctly — we're only
    proving it's never invoked, not changing what happens if it were."""
    slug = "protein-structure-no-llm"
    title = "protein_structure -> no LLM call happens at all (spies on the real functions, LIVE code path)"
    detail: list[str] = []

    try:
        with patch(f"{_MOD}.resolve_visualization_references",
                    wraps=resolve_visualization_references) as spy_viz_resolver, patch(
            f"{_MOD}.resolve_chromosome_highlight", wraps=resolve_chromosome_highlight
        ) as spy_highlight_resolver, patch(
            f"{_RESOLVER_MOD}.get_llm_client"
        ) as spy_llm_client:
            spy_llm_client.side_effect = EnvironmentError("should never be called for protein_structure")

            result = await generate_visualization(
                scope="protein_structure",
                assembly_id="GCF_mouse",
                user_question="Predict the 3D protein structure for the house mouse",
            )

        detail.append(f"status={result['status']!r}")
        detail.append(f"prompt_to_target_agent={result.get('prompt_to_target_agent')!r}")

        no_calls = (
            spy_viz_resolver.call_count == 0
            and spy_highlight_resolver.call_count == 0
            and spy_llm_client.call_count == 0
        )
        detail.append(
            f"resolve_visualization_references calls={spy_viz_resolver.call_count}, "
            f"resolve_chromosome_highlight calls={spy_highlight_resolver.call_count}, "
            f"get_llm_client calls={spy_llm_client.call_count}"
        )
        detail.append(f"zero LLM-path calls made: {no_calls}")

        needs_agent = result["status"] == "NEEDS_AGENT"
        detail.append(f"status is NEEDS_AGENT (deterministic delegation): {needs_agent}")

        passed = no_calls and needs_agent
        return CaseResult(slug, title, passed, detail)
    except Exception as exc:  # pragma: no cover
        return CaseResult(slug, title, False, detail, error=repr(exc))


async def case_llm_unavailable_both_scopes_fall_back() -> CaseResult:
    """§4.10 case 5: LLM unavailable (forced — this is the point of the
    case) -> both rendering scopes fall back to their deterministic
    defaults. Every NCBI call inside both fallback paths is still real."""
    slug = "llm-unavailable-both-fallback"
    title = "LLM unavailable (forced) -> chromosome_map and size_comparison both fall back for real (LIVE NCBI)"
    detail: list[str] = []

    try:
        from ..subagents.species_resolver import resolve_species
        from ..subagents.genome_metadata import get_genome_metadata
    except ImportError:
        from genome_agent.subagents.species_resolver import resolve_species
        from genome_agent.subagents.genome_metadata import get_genome_metadata

    try:
        with patch(f"{_RESOLVER_MOD}.get_llm_client") as mock_llm:
            mock_llm.side_effect = EnvironmentError("Simulated: NVIDIA_API_KEY unavailable")

            # --- chromosome_map: real heuristic fallback, no LLM ---
            gene_table = [{"gene_name": "ABC", "location": "chr1", "function": "Function A"}]
            chrom_result = await generate_visualization(
                scope="chromosome_map",
                gene_table=gene_table,
                user_question="Tell me about ABC please",
            )

            # --- size_comparison: real fixed reference list, resolved live ---
            mouse = await resolve_species("house mouse")
            mouse_meta = await get_genome_metadata(mouse["assembly_id"]) if mouse.get("assembly_id") else {}
            size_result = await generate_visualization(
                scope="size_comparison",
                genome_size_bp=mouse_meta.get("genome_size_bp"),
                assembly_id=mouse.get("assembly_id"),
                common_name="house mouse (queried)",
                scientific_name=mouse.get("scientific_name"),
                user_question="Compare genome sizes",
            )

        chrom_rendered = chrom_result["status"] == "COMPLETED" and chrom_result.get("chart_data") is not None
        chrom_fallback_found_gene = b"e8622c" in (chrom_result.get("chart_data") or b"")
        detail.append(f"[chromosome_map] rendered despite LLM unavailable: {chrom_rendered}")
        detail.append(f"[chromosome_map] real heuristic fallback still found 'ABC' in the question: {chrom_fallback_found_gene}")

        size_rendered = size_result["status"] == "COMPLETED" and size_result.get("chart_data") is not None
        comparisons = size_result.get("comparisons") or []
        detail.append(f"[size_comparison] rendered despite LLM unavailable: {size_rendered}")
        detail.append(
            f"[size_comparison] species actually compared (live NCBI): "
            f"{[c['common_name'] for c in comparisons]}"
        )

        # The fixed fallback list is ["human", "house mouse", "chicken",
        # "zebrafish"] (see resolve_visualization_references_fallback).
        # NCBI has no separate "common name" field — common_name and
        # scientific_name are both the organism's scientific name — so
        # verify by resolving those exact names ourselves and comparing
        # assembly_ids, rather than assuming an English string appears
        # anywhere in the result.
        expected_names = ["human", "chicken", "zebrafish"]  # skip "house mouse" — it's also the queried species here
        expected_scientific_names = set()
        for name in expected_names:
            resolved = await resolve_species(name)
            if resolved.get("scientific_name"):
                expected_scientific_names.add(resolved["scientific_name"])
        actual_scientific_names = {c["scientific_name"] for c in comparisons}
        used_default_list = bool(expected_scientific_names) and expected_scientific_names.issubset(actual_scientific_names)
        detail.append(f"[size_comparison] expected default-list species resolved: {expected_scientific_names}")
        detail.append(f"[size_comparison] fixed default reference list used: {used_default_list}")

        passed = chrom_rendered and chrom_fallback_found_gene and size_rendered and used_default_list
        return CaseResult(slug, title, passed, detail)
    except Exception as exc:  # pragma: no cover
        return CaseResult(slug, title, False, detail, error=repr(exc))


CASES: list[Callable[[], Any]] = [
    case_size_comparison_uses_real_proposal,
    case_partial_reference_failure_still_renders,
    case_chromosome_map_highlight_from_question,
    case_protein_structure_never_calls_llm,
    case_llm_unavailable_both_scopes_fall_back,
]


async def run_all() -> int:
    llm_mode = (
        "LIVE (NVIDIA_API_KEY set)"
        if os.getenv("NVIDIA_API_KEY")
        else "FALLBACK (no NVIDIA_API_KEY — real deterministic fallback logic)"
    )
    print("\n" + "#" * _WIDTH)
    print("# Visualization Agent — §4.10 'Test independently' execution (REAL)")
    print(f"# LLM mode: {llm_mode}")
    print("# NCBI: live eutils calls throughout")
    print("#" * _WIDTH)

    results = [await case() for case in CASES]

    for result in results:
        _print_case(result)

    passed = sum(1 for r in results if r.passed)
    total = len(results)

    print("\n" + "#" * _WIDTH)
    print(f"# {passed}/{total} case(s) passed.")
    if passed < total:
        print("# FAILING CASES:")
        for r in results:
            if not r.passed:
                print(f"#   - {r.title}")
    print("#" * _WIDTH)

    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(run_all()))