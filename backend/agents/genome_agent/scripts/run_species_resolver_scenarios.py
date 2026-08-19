"""
Live scenario runner for the Species Resolver subagent (spec §10).

Unlike tests/test_species_resolver_llm.py (which mocks both the LLM client
AND the NCBI-calling tools for fast, deterministic unit tests), this is a
plain executable script meant to be run by hand against the real world:

    - Real LLM calls through get_llm_client() (NVIDIA_API_KEY required).
    - Real NCBI Taxonomy/Assembly eutils calls through search_taxonomy /
      search_assembly_by_taxid — no NCBI mocking anywhere in this file.

It covers the five scenarios from the task guide's §10 exactly:

    1. 3-candidate ambiguous case ("elephant")     -> picks one, confidence < 1.0
    2. Clean single-match case ("house mouse")     -> confidence == 1.0, reasoning present
    3. Empty-then-reformulated-match case (typo)   -> exactly one retry
    4. LLM client raises                           -> deterministic fallback still works
    5. Grounding: fabricated assembly_id           -> rejected, loop retries, real id wins

Honesty note on scenario 5: a well-behaved live model won't reliably decide
to hallucinate an assembly_id on demand, so an unscripted live run could
"pass" scenario 5 for the wrong reason (the rejection branch never fires).
To make the test actually prove the rejection path works, scenario 5 scripts
*only* the model's decisions (which tool to call, what to submit) via a
stand-in client — search_taxonomy and search_assembly_by_taxid are still the
real tool objects making real eutils calls. This is called out again right
before it runs. Every other scenario is 100% live, nothing scripted.

Because scenarios 2 and 3 assert on live LLM judgment (an exact confidence
value, an exact retry count), a single run can legitimately come back WARN
instead of PASS without indicating a bug — that's noted inline.

Run every scenario:
    python -m gene_agent.scripts.run_species_resolver_scenarios

Run just one:
    python -m gene_agent.scripts.run_species_resolver_scenarios --scenario house-mouse

List scenario slugs:
    python -m gene_agent.scripts.run_species_resolver_scenarios --list

or, from inside the container:
    python scripts/run_species_resolver_scenarios.py
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
import time
from typing import Any, Awaitable, Callable
from unittest.mock import MagicMock

# Node-level INFO logs from the subagent are useful in isolation but drown
# out the trace below.
logging.getLogger().setLevel(logging.WARNING)

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


# ---------------------------------------------------------------------------
# Scenario 1 — ambiguous common name
# ---------------------------------------------------------------------------
async def scenario_ambiguous_elephant() -> dict:
    _header("1. Ambiguous common name — 'elephant' (multiple taxonomy candidates)")
    t0 = time.monotonic()
    result = await sr.resolve_species_llm("elephant")
    elapsed = time.monotonic() - t0

    if result is None:
        return _result(
            "ambiguous-elephant", "SKIP",
            "resolve_species_llm returned None (no NVIDIA_API_KEY set, or the "
            "live endpoint is unavailable right now)",
        )

    _kv("assembly_id", result.get("assembly_id"))
    _kv("scientific_name", result.get("scientific_name"))
    _kv("common_name", result.get("common_name"))
    _kv("confidence", result.get("confidence"))
    _kv("reasoning", result.get("reasoning"))
    _kv("elapsed", f"{elapsed:.2f}s")

    if result.get("assembly_id") is None:
        return _result("ambiguous-elephant", "FAIL", "assembly_id is None — expected a pick, not a dead end")
    if not (result.get("confidence", 1.0) < 1.0):
        return _result("ambiguous-elephant", "FAIL", f"confidence={result.get('confidence')} — expected < 1.0 for an ambiguous case")
    return _result("ambiguous-elephant", "PASS")


# ---------------------------------------------------------------------------
# Scenario 2 — clean single match
# ---------------------------------------------------------------------------
async def scenario_house_mouse() -> dict:
    _header("2. Clean single match — 'house mouse'")
    t0 = time.monotonic()
    result = await sr.resolve_species_llm("house mouse")
    elapsed = time.monotonic() - t0

    if result is None:
        return _result("house-mouse", "SKIP", "resolve_species_llm returned None — no API key / endpoint unavailable")

    _kv("assembly_id", result.get("assembly_id"))
    _kv("confidence", result.get("confidence"))
    _kv("reasoning", result.get("reasoning"))
    _kv("elapsed", f"{elapsed:.2f}s")

    if not result.get("reasoning"):
        return _result("house-mouse", "FAIL", "reasoning is empty")
    if result.get("confidence") == 1.0:
        return _result("house-mouse", "PASS")
    return _result(
        "house-mouse", "WARN",
        f"confidence={result.get('confidence')} (spec expects exactly 1.0 for an "
        "unambiguous match — live model judgment can vary run to run; not "
        "necessarily a bug)",
    )


# ---------------------------------------------------------------------------
# Scenario 3 — empty search, one reformulation
# ---------------------------------------------------------------------------
async def scenario_reformulation_retry() -> dict:
    _header("3. Empty-then-reformulated — misspelled species name")

    queries_attempted: list[str] = []
    real_search_taxonomy_core = sr._search_taxonomy_core

    async def _spy(query: str):
        # Spy, not a mock: records the query, then calls straight through to
        # the real NCBI-hitting implementation.
        queries_attempted.append(query)
        return await real_search_taxonomy_core(query)

    sr._search_taxonomy_core = _spy
    try:
        t0 = time.monotonic()
        result = await sr.resolve_species_llm("asain elefant")
        elapsed = time.monotonic() - t0
    finally:
        sr._search_taxonomy_core = real_search_taxonomy_core

    _kv("queries attempted", queries_attempted)
    _kv("elapsed", f"{elapsed:.2f}s")
    if result is not None:
        _kv("assembly_id", result.get("assembly_id"))
        _kv("confidence", result.get("confidence"))

    if len(queries_attempted) == 0:
        return _result("reformulation-retry", "SKIP", "LLM never called search_taxonomy — no API key / endpoint unavailable")
    if len(queries_attempted) == 1:
        return _result(
            "reformulation-retry", "WARN",
            f"only 1 search_taxonomy call ({queries_attempted[0]!r}) — either the "
            "misspelling resolved on the first try, or the model gave up without "
            "reformulating",
        )
    if len(queries_attempted) == 2:
        return _result("reformulation-retry", "PASS", f"exactly one retry: {queries_attempted}")
    return _result(
        "reformulation-retry", "FAIL",
        f"{len(queries_attempted)} search_taxonomy calls — spec says try ONE "
        f"reformulation, not many: {queries_attempted}",
    )


# ---------------------------------------------------------------------------
# Scenario 4 — LLM unavailable, deterministic fallback
# ---------------------------------------------------------------------------
async def scenario_llm_raises_fallback() -> dict:
    _header("4. LLM client unavailable — deterministic NCBI fallback")

    saved_key = os.environ.pop("NVIDIA_API_KEY", None)
    try:
        llm_result = await sr.resolve_species_llm("tiger")
        _kv("resolve_species_llm() with no key", llm_result)
        if llm_result is not None:
            return _result("llm-raises-fallback", "FAIL", "resolve_species_llm returned a result even with no API key set")

        t0 = time.monotonic()
        fallback_result = await sr.resolve_species("tiger")
        elapsed = time.monotonic() - t0
    finally:
        if saved_key is not None:
            os.environ["NVIDIA_API_KEY"] = saved_key

    _kv("fallback assembly_id", fallback_result.get("assembly_id"))
    _kv("fallback confidence", fallback_result.get("confidence"))
    _kv("fallback reasoning", fallback_result.get("reasoning"))
    _kv("elapsed", f"{elapsed:.2f}s")

    if fallback_result.get("assembly_id") is None:
        return _result("llm-raises-fallback", "FAIL", "deterministic fallback also returned no assembly_id")
    if fallback_result.get("confidence") != 0.5:
        return _result("llm-raises-fallback", "WARN", f"fallback confidence={fallback_result.get('confidence')} (spec example uses 0.5)")
    return _result("llm-raises-fallback", "PASS")


# ---------------------------------------------------------------------------
# Scenario 5 — grounding: fabricated assembly_id must be rejected
# ---------------------------------------------------------------------------
async def scenario_grounding_fabricated_id() -> dict:
    _header("5. Grounding — LLM tries to submit a fabricated assembly_id")
    print(
        "  NOTE: the only scripted scenario in this file. A well-behaved live\n"
        "  model won't reliably choose to hallucinate an id on demand, so the\n"
        "  model's *decisions* are scripted here to guarantee the rejection path\n"
        "  actually fires. search_taxonomy and search_assembly_by_taxid are the\n"
        "  real tool objects below and still make real eutils calls — only\n"
        "  client.invoke() (i.e. 'what would the model do next') is stubbed."
    )

    fabricated_id = "GCF_FAKE_999999999.1"
    captured_assembly_results: list[dict] = []
    # Spy on the plain async core function, NOT sr.search_assembly_by_taxid
    # itself: that's a langchain @tool-wrapped StructuredTool, which is a
    # pydantic v2 BaseModel under the hood. Newer pydantic/langchain-core
    # reject assigning arbitrary attributes (like .ainvoke) onto a
    # BaseModel instance ("object has no field ..."), so patching the tool
    # object directly breaks. Patching the underlying _core function is
    # exactly what scenario 3 (reformulation-retry) already does for
    # _search_taxonomy_core, for the same reason — the @tool wrapper calls
    # the module-level _core function by name at call time, so replacing
    # the module attribute is visible to it without touching the tool.
    real_search_assembly_by_taxid_core = sr._search_assembly_by_taxid_core

    async def _assembly_spy(tax_id):
        # Spy, not a mock: the real tool still runs and hits live NCBI. We
        # just also keep a copy of what it returned so the scripted "model"
        # below can submit a real, grounded id afterwards.
        result = await real_search_assembly_by_taxid_core(tax_id)
        captured_assembly_results.clear()
        if isinstance(result, list):
            captured_assembly_results.extend(result)
        return result

    def _tool_call(name: str, args: dict, call_id: str) -> dict:
        return {"id": call_id, "name": name, "args": args, "type": "tool_call"}

    def _response(tool_calls: list[dict]):
        r = MagicMock()
        r.tool_calls = tool_calls
        r.content = ""
        return r

    step_counter = {"n": 0}

    def _scripted_invoke(_messages):
        step_counter["n"] += 1
        step = step_counter["n"]

        if step == 1:
            # "Model" decides to look up the taxon first (real NCBI call).
            return _response([_tool_call("search_taxonomy", {"query": "tiger"}, "c1")])
        if step == 2:
            # "Model" hallucinates a final answer before ever calling
            # search_assembly_by_taxid — nothing in the grounding record
            # supports this id at all.
            return _response([
                _tool_call(
                    "SpeciesResolverOutput",
                    {
                        "assembly_id": fabricated_id,
                        "scientific_name": "Panthera tigris",
                        "common_name": "tiger",
                        "confidence": 1.0,
                        "reasoning": "fabricated for grounding test",
                    },
                    "c2",
                )
            ])
        if step == 3:
            # After rejection, "model" does the right thing: looks up real
            # assemblies for the tiger taxon (well-known NCBI tax_id 9694).
            return _response([_tool_call("search_assembly_by_taxid", {"tax_id": "9694"}, "c3")])
        # step >= 4: submit whatever real assembly_id the real call in step 3
        # actually returned.
        if not captured_assembly_results:
            return _response([])
        real_id = captured_assembly_results[0].get("assembly_id")
        return _response([
            _tool_call(
                "SpeciesResolverOutput",
                {
                    "assembly_id": real_id,
                    "scientific_name": captured_assembly_results[0].get("scientific_name", "Panthera tigris"),
                    "common_name": "tiger",
                    "confidence": 1.0,
                    "reasoning": "corrected to a grounded id after rejection",
                },
                "c4",
            )
        ])

    client = MagicMock()
    client.bind_tools.return_value = client
    client.invoke = _scripted_invoke

    real_get_llm_client = sr.get_llm_client
    sr.get_llm_client = lambda: client
    sr._search_assembly_by_taxid_core = _assembly_spy
    try:
        t0 = time.monotonic()
        result = await sr.resolve_species_llm("tiger")
        elapsed = time.monotonic() - t0
    finally:
        sr.get_llm_client = real_get_llm_client
        sr._search_assembly_by_taxid_core = real_search_assembly_by_taxid_core

    _kv("result", result)
    _kv("real assemblies seen", captured_assembly_results)
    _kv("elapsed", f"{elapsed:.2f}s")

    if result is None:
        return _result("grounding-fabricated", "FAIL", "loop gave up entirely instead of retrying after rejection")
    if result.get("assembly_id") == fabricated_id:
        return _result("grounding-fabricated", "FAIL", "fabricated assembly_id was accepted — grounding check did not fire")
    if not result.get("assembly_id"):
        return _result("grounding-fabricated", "FAIL", "no assembly_id returned at all")
    return _result("grounding-fabricated", "PASS", f"fabricated id rejected; real id '{result['assembly_id']}' accepted instead")


SCENARIOS: dict[str, Callable[[], Awaitable[dict]]] = {
    "ambiguous-elephant": scenario_ambiguous_elephant,
    "house-mouse": scenario_house_mouse,
    "reformulation-retry": scenario_reformulation_retry,
    "llm-raises-fallback": scenario_llm_raises_fallback,
    "grounding-fabricated": scenario_grounding_fabricated_id,
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
        llm_mode = "NO KEY SET — every non-fallback scenario below will SKIP"
    print("\n" + "#" * _WIDTH)
    print("# Species Resolver — Live Scenario Run (real NCBI, real LLM)")
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
        help="Run only the named scenario(s) instead of all five. Use --list for slugs.",
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
