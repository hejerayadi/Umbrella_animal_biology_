"""
Live scenario runner for the Genome Metadata subagent (spec §10).

Unlike tests/test_genome_metadata_llm.py (which mocks both the LLM client
AND the NCBI-calling tools), this is a plain executable script meant to be
run by hand against the real world:

    - Real LLM calls through get_llm_client() (NVIDIA_API_KEY required).
    - Real NCBI Assembly eutils calls through fetch_assembly_stats /
      list_alternate_assemblies — no NCBI mocking, except scenario 4 below,
      which corrupts exactly one already-real response (see its note).

It covers the four scenarios from the task guide's §10:

    1. Single-assembly case              -> no substitution occurs
    2. Multi-assembly, better option     -> substitution happens and is explained
    3. LLM unavailable                   -> deterministic single-fetch fallback still works
    4. Garbage total_length              -> surfaces as null, not as a number

Honesty notes:
  - Scenario 1 picks a species whose current assembly is already the best
    RefSeq record NCBI has, so "no substitution" is the *expected* live
    outcome — but it's still the model's live judgment call, so a
    substitution-anyway result is reported as WARN, not FAIL.
  - Scenario 2 needs a species that actually *has* an inferior alternate
    assembly on record right now. That's a live-data precondition this
    script can't guarantee, so it searches a short list of species known to
    have historical draft assemblies and SKIPs cleanly if none of them
    currently qualify, rather than reporting a false FAIL.
  - Scenario 4 needs NCBI to hand back a corrupted total_length, which NCBI
    will obviously never do on purpose. To test the real defensive parsing
    code (not a reimplementation of it) without inventing a payload from
    thin air, this scenario makes a real esearch + esummary round trip and
    then corrupts only the total_length digits inside the real returned XML
    before handing it to the real parser. This is the one non-fully-live
    step in this file and is called out again at the point it runs.

Run every scenario:
    python -m gene_agent.scripts.run_genome_metadata_scenarios

Run just one:
    python -m gene_agent.scripts.run_genome_metadata_scenarios --scenario llm-unavailable

List scenario slugs:
    python -m gene_agent.scripts.run_genome_metadata_scenarios --list

or, from inside the container:
    python scripts/run_genome_metadata_scenarios.py
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
import re
import sys
import time
from typing import Any, Awaitable, Callable

logging.getLogger().setLevel(logging.WARNING)

from ..subagents import genome_metadata as gm
from ..subagents import species_resolver as sr

_WIDTH = 72


def _header(title: str) -> None:
    print()
    print("=" * _WIDTH)
    print(title)
    print("=" * _WIDTH)


def _kv(label: str, value: Any) -> None:
    print(f"  {label:<24}: {value}")


def _result(name: str, status: str, detail: str = "") -> dict:
    line = f"  >> {status}: {name}"
    if detail:
        line += f" — {detail}"
    print(line)
    return {"name": name, "status": status, "detail": detail}


async def _resolve_tax_id(species_name: str) -> str | None:
    """Real NCBI Taxonomy lookup, reusing the species resolver's own core
    function (no LLM needed for this — we just need a tax_id to work with)."""
    candidates = await sr._search_taxonomy_core(species_name)
    if not candidates:
        return None
    return str(candidates[0].get("tax_id")) or None


# ---------------------------------------------------------------------------
# Scenario 1 — single/best assembly, no substitution expected
# ---------------------------------------------------------------------------
async def scenario_single_assembly_no_substitution() -> dict:
    _header("1. Single-assembly case — 'house mouse' (already the best RefSeq record)")

    species_result = await sr.resolve_species("house mouse")
    given_id = species_result.get("assembly_id")
    if not given_id:
        return _result("single-no-substitution", "SKIP", "could not resolve house mouse's assembly_id live")

    _kv("resolved assembly_id", given_id)

    t0 = time.monotonic()
    result = await gm.resolve_metadata_llm("house mouse", given_id)
    elapsed = time.monotonic() - t0

    if result is None:
        return _result("single-no-substitution", "SKIP", "resolve_metadata_llm returned None — no NVIDIA_API_KEY set / endpoint unavailable")

    _kv("genome_size_bp", result.get("genome_size_bp"))
    _kv("chromosome_count", result.get("chromosome_count"))
    _kv("assembly_level", result.get("assembly_level"))
    _kv("assembly_id_used", result.get("assembly_id_used"))
    _kv("reasoning", result.get("reasoning"))
    _kv("elapsed", f"{elapsed:.2f}s")

    if result.get("assembly_id_used") == given_id:
        return _result("single-no-substitution", "PASS")
    return _result(
        "single-no-substitution", "WARN",
        f"model substituted {given_id!r} -> {result.get('assembly_id_used')!r}. "
        "That's a live judgment call; check reasoning above to see if it's justified "
        "(e.g. a newer RefSeq assembly really was published since this script was written).",
    )


# ---------------------------------------------------------------------------
# Scenario 2 — multi-assembly, expect substitution to the better one
# ---------------------------------------------------------------------------
_SUBSTITUTION_CANDIDATE_SPECIES = ["zebrafish", "chicken", "cow", "Atlantic salmon"]


async def _find_inferior_assembly() -> tuple[str, str, str] | None:
    """Search a short list of species for one that currently has a
    non-RefSeq (GCA_) assembly on record alongside a RefSeq (GCF_) one.
    Returns (species_name, tax_id, gca_assembly_id) or None if none of the
    candidates currently qualify — this is real, live NCBI data, so it can
    change out from under us."""
    for species_name in _SUBSTITUTION_CANDIDATE_SPECIES:
        tax_id = await _resolve_tax_id(species_name)
        if not tax_id:
            continue
        alternates = await gm.list_alternate_assemblies.ainvoke({"tax_id": tax_id})
        gca_ids = [a["assembly_id"] for a in alternates if a.get("assembly_id", "").startswith("GCA_")]
        gcf_ids = [a["assembly_id"] for a in alternates if a.get("assembly_id", "").startswith("GCF_")]
        if gca_ids and gcf_ids:
            return species_name, tax_id, gca_ids[0]
    return None


async def scenario_multi_assembly_substitution() -> dict:
    _header("2. Multi-assembly case — expect substitution to a better RefSeq assembly")

    found = await _find_inferior_assembly()
    if found is None:
        return _result(
            "multi-assembly-substitution", "SKIP",
            f"none of {_SUBSTITUTION_CANDIDATE_SPECIES} currently has both a GCA_ and a "
            "GCF_ assembly on record — this is a live-data precondition, not a code issue",
        )

    species_name, tax_id, given_id = found
    _kv("species", species_name)
    _kv("tax_id", tax_id)
    _kv("given (inferior) assembly_id", given_id)

    t0 = time.monotonic()
    result = await gm.resolve_metadata_llm(species_name, given_id)
    elapsed = time.monotonic() - t0

    if result is None:
        return _result("multi-assembly-substitution", "SKIP", "resolve_metadata_llm returned None — no NVIDIA_API_KEY set / endpoint unavailable")

    _kv("assembly_id_used", result.get("assembly_id_used"))
    _kv("assembly_level", result.get("assembly_level"))
    _kv("reasoning", result.get("reasoning"))
    _kv("elapsed", f"{elapsed:.2f}s")

    if result.get("assembly_id_used") == given_id:
        return _result("multi-assembly-substitution", "FAIL", "model kept the inferior GCA_ assembly despite a GCF_ alternative being available")
    if not result.get("assembly_id_used", "").startswith("GCF_"):
        return _result("multi-assembly-substitution", "WARN", f"substituted to {result.get('assembly_id_used')!r}, which isn't RefSeq — worth a look")
    if not result.get("reasoning"):
        return _result("multi-assembly-substitution", "WARN", "substitution happened but reasoning is empty")
    return _result("multi-assembly-substitution", "PASS", f"substituted {given_id} -> {result['assembly_id_used']}")


# ---------------------------------------------------------------------------
# Scenario 3 — LLM unavailable, deterministic single-fetch fallback
# ---------------------------------------------------------------------------
async def scenario_llm_unavailable_fallback() -> dict:
    _header("3. LLM unavailable — deterministic single-fetch fallback")

    species_result = await sr.resolve_species("tiger")
    given_id = species_result.get("assembly_id")
    if not given_id:
        return _result("llm-unavailable-fallback", "SKIP", "could not resolve tiger's assembly_id live")

    _kv("assembly_id", given_id)

    saved_key = os.environ.pop("NVIDIA_API_KEY", None)
    try:
        llm_result = await gm.resolve_metadata_llm("tiger", given_id)
        _kv("resolve_metadata_llm() with no key", llm_result)
        if llm_result is not None:
            return _result("llm-unavailable-fallback", "FAIL", "resolve_metadata_llm returned a result even with no API key set")

        t0 = time.monotonic()
        fallback_result = await gm.fetch_metadata_fallback(given_id)
        elapsed = time.monotonic() - t0
    finally:
        if saved_key is not None:
            os.environ["NVIDIA_API_KEY"] = saved_key

    _kv("fallback genome_size_bp", fallback_result.get("genome_size_bp"))
    _kv("fallback chromosome_count", fallback_result.get("chromosome_count"))
    _kv("fallback assembly_id_used", fallback_result.get("assembly_id_used"))
    _kv("fallback reasoning", fallback_result.get("reasoning"))
    _kv("elapsed", f"{elapsed:.2f}s")

    if fallback_result.get("assembly_id_used") != given_id:
        return _result("llm-unavailable-fallback", "FAIL", "fallback must never substitute — assembly_id_used changed")
    if fallback_result.get("genome_size_bp") is None:
        return _result("llm-unavailable-fallback", "WARN", "fallback ran without crashing, but genome_size_bp came back null — check reasoning above")
    return _result("llm-unavailable-fallback", "PASS")


# ---------------------------------------------------------------------------
# Scenario 4 — garbage total_length surfaces as null
# ---------------------------------------------------------------------------
class _CorruptedResponse:
    """Wraps a real requests.Response, replacing only its .json() payload."""

    def __init__(self, real_response, corrupted_json: dict):
        self._real_response = real_response
        self._corrupted_json = corrupted_json
        self.status_code = real_response.status_code

    def json(self):
        return self._corrupted_json

    def raise_for_status(self):
        return None


def _corrupt_total_length(meta_xml: str) -> tuple[str, bool]:
    """Replace the numeric total_length inside a real <Stats> XML blob with
    a non-numeric value. Returns (corrupted_xml, did_replace)."""
    pattern = re.compile(r'(category="total_length"[^>]*>)\s*\d+')
    corrupted, count = pattern.subn(r"\1NOT_A_NUMBER", meta_xml, count=1)
    if count == 0:
        # Live XML shape didn't match what we expected — fall back to a
        # purpose-built garbage <Stats> block so the scenario can still run
        # deterministically rather than silently skipping the check.
        corrupted = '<Stats><Stat category="total_length" sequence_tag="all">NOT_A_NUMBER</Stat></Stats>'
        return corrupted, False
    return corrupted, True


async def scenario_garbage_total_length() -> dict:
    _header("4. Garbage total_length — must surface as null, not a number")
    print(
        "  NOTE: the one non-fully-live scenario in this file. NCBI will never\n"
        "  intentionally return a corrupted total_length, so this makes a real\n"
        "  esearch+esummary round trip and then corrupts only the total_length\n"
        "  digits inside the real returned XML before handing it to the real\n"
        "  parser (_parse_meta_stats) and the real fetch_assembly_stats tool."
    )

    species_result = await sr.resolve_species("tiger")
    assembly_id = species_result.get("assembly_id")
    if not assembly_id:
        return _result("garbage-total-length", "SKIP", "could not resolve tiger's assembly_id live")

    _kv("assembly_id", assembly_id)

    real_ncbi_get = gm.ncbi_get
    replaced_for_real = {"value": None}

    def _corrupting_ncbi_get(params, **kwargs):
        is_summary = params.get("path") == "esummary.fcgi"
        resp = real_ncbi_get(params, **kwargs)
        if not is_summary:
            return resp
        data = resp.json()
        result_block = data.get("result", {})
        uids = result_block.get("uids") or [k for k in result_block.keys() if k != "uids"]
        if not uids:
            return resp
        uid = uids[0]
        info = dict(result_block.get(uid, {}))
        corrupted_xml, did_replace = _corrupt_total_length(info.get("meta", ""))
        replaced_for_real["value"] = did_replace
        info["meta"] = corrupted_xml
        new_result_block = dict(result_block)
        new_result_block[uid] = info
        corrupted_data = dict(data)
        corrupted_data["result"] = new_result_block
        return _CorruptedResponse(resp, corrupted_data)

    gm.ncbi_get = _corrupting_ncbi_get
    try:
        t0 = time.monotonic()
        result = await gm.fetch_assembly_stats.ainvoke({"assembly_id": assembly_id})
        elapsed = time.monotonic() - t0
    finally:
        gm.ncbi_get = real_ncbi_get

    _kv("corrupted a real payload", replaced_for_real["value"])
    _kv("genome_size_bp", result.get("genome_size_bp") if isinstance(result, dict) else result)
    _kv("elapsed", f"{elapsed:.2f}s")

    if not isinstance(result, dict):
        return _result("garbage-total-length", "FAIL", f"fetch_assembly_stats raised or returned non-dict: {result!r}")
    if result.get("genome_size_bp") is not None:
        return _result("garbage-total-length", "FAIL", f"genome_size_bp={result.get('genome_size_bp')!r} — garbage value leaked through instead of becoming null")
    return _result("garbage-total-length", "PASS", "corrupted total_length correctly surfaced as null")


SCENARIOS: dict[str, Callable[[], Awaitable[dict]]] = {
    "single-no-substitution": scenario_single_assembly_no_substitution,
    "multi-assembly-substitution": scenario_multi_assembly_substitution,
    "llm-unavailable": scenario_llm_unavailable_fallback,
    "garbage-total-length": scenario_garbage_total_length,
}


def _print_summary(results: list[dict]) -> None:
    _header("Summary")
    for r in results:
        print(f"  {r['status']:<5} {r['name']}")
    counts = {"PASS": 0, "WARN": 0, "FAIL": 0, "SKIP": 0}
    for r in results:
        counts[r["status"]] += 1
    print()
    print(f"  {counts['PASS']} passed, {counts['WARN']} warned, {counts['FAIL']} failed, {counts['SKIP']} skipped")


async def run(selected: list[str] | None) -> int:
    if os.getenv("NVIDIA_API_KEY"):
        llm_mode = "LIVE (NVIDIA_API_KEY set)"
    else:
        llm_mode = "NO KEY SET — every LLM-path scenario below will SKIP"
    print("\n" + "#" * _WIDTH)
    print("# Genome Metadata — Live Scenario Run (real NCBI, real LLM)")
    print(f"# LLM mode: {llm_mode}")
    print("#" * _WIDTH)

    slugs = selected or list(SCENARIOS.keys())
    results = []
    for slug in slugs:
        results.append(await SCENARIOS[slug]())

    _print_summary(results)
    return 1 if any(r["status"] == "FAIL" for r in results) else 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scenario", "-s",
        nargs="+",
        metavar="SLUG",
        choices=list(SCENARIOS.keys()),
        help="Run only the named scenario(s) instead of all four. Use --list for slugs.",
    )
    parser.add_argument("--list", "-l", action="store_true", help="Print available scenario slugs and exit.")
    args = parser.parse_args()

    if args.list:
        for slug in SCENARIOS:
            print(slug)
        return

    exit_code = asyncio.run(run(args.scenario))
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
