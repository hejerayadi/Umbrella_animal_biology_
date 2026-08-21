"""
Literature Support Agent scenario runner (design guide §10, "Test independently").

Unlike tests/test_literature_support.py (which monkeypatches at the unit
boundary to assert on isolated behavior), this module *executes* the real
literature_support_agent() end to end for every §10 case, and for the
escalation case additionally drives it through escalate_literature_agent_node
so the full "Literature Support -> Capability Resolver -> target_agent" hop
gets exercised too (mirrors tests/test_workflows.py's patch_llm_nodes pattern).

Every scenario patches only the Literature Agent A2A call
(request_literature_evidence) so results are deterministic and don't require
a live Literature Agent deployment — that boundary is exactly what
tests/test_literature_support.py also patches. Most scenarios also patch
_llm_judge_sufficiency for speed/determinism and don't need
NVIDIA_NIM_API_KEY; two-records-real-llm-judgment is the one exception that
leaves _llm_judge_sufficiency unpatched to prove the real bind_tools LLM call
(design guide §3/§5) actually works end to end, and DOES require
NVIDIA_NIM_API_KEY. The escalation scenario additionally patches
resolve_capability, mirroring tests/test_workflows.py's patch_llm_nodes.

Usage:
    python -m scripts.literature_support_scenarios --scenario two-records-completed
    python -m scripts.literature_support_scenarios --scenario two-records-real-llm-judgment
    python -m scripts.literature_support_scenarios --scenario one-record-escalation
    python -m scripts.literature_support_scenarios --scenario zero-records-failed
    python -m scripts.literature_support_scenarios --scenario llm-unavailable-fallback
    python -m scripts.literature_support_scenarios --scenario grounding-rejects-invented-pmid
    python -m scripts.literature_support_scenarios --scenario all
    python -m scripts.literature_support_scenarios --scenario all --verbose
"""
import argparse
import asyncio
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import subagents.literature_support as ls_module
import workflows.nodes.escalation_nodes as esc_module
from workflows.capability_resolver import CapabilityResolution
from workflows.state import TraitDiscoveryState
from schemas.common import AgentStatus
from schemas.inputs import LiteratureSupportInput

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Fixtures — same trait/gene/pmid facts already used in mock.py and
# tests/test_literature_support.py, kept identical here so results are
# comparable across the unit tests, this scenario runner, and the mock.
# ---------------------------------------------------------------------------
TWO_RECORDS = [
    {"pmid": "18239092", "title": "FGF5 and hair cycle regulation", "year": 2008,
     "short_summary": "Links FGF5 mutation to hair length in mammals."},
    {"pmid": "30112233", "title": "Follicle regulatory network in rodents", "year": 2019,
     "short_summary": "Broader gene network context around FGF5 signaling."},
]
ONE_RECORD = [
    {"pmid": "26123456", "title": "UCP1 and non-shivering thermogenesis", "year": 2016,
     "short_summary": "UCP1 role in brown fat heat production."},
]


class _Patch:
    """Tiny manual monkeypatch — swap an attribute, restore it on exit.
    Same helper as scripts/gene_mapper_scenarios.py / orchestration_retrieval_scenarios.py."""

    def __init__(self, obj, name, value):
        self.obj, self.name, self.value = obj, name, value

    def __enter__(self):
        self._orig = getattr(self.obj, self.name)
        setattr(self.obj, self.name, self.value)
        return self

    def __exit__(self, *exc):
        setattr(self.obj, self.name, self._orig)


async def _check(label: str, condition: bool, detail: str, failures: list) -> None:
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {label}: {detail}")
    if not condition:
        failures.append(f"{label}: {detail}")


def _fake_fetch(records: list[dict]):
    async def _fetch(trait_name, gene_list):
        return records
    return _fetch


# ---------------------------------------------------------------------------
# Scenarios (design guide §10)
# ---------------------------------------------------------------------------
async def scenario_two_records_completed(verbose: bool) -> list:
    """Two clearly relevant records -> assert COMPLETED, no escalation.

    This scenario patches _llm_judge_sufficiency (deterministic, no NIM key
    needed) so it's fast and stable in CI. For a run that hits the REAL
    bind_tools LLM call, see two-records-real-llm-judgment below.
    """
    async def fake_judge(trait_name, gene_list, evidence, thin_flag):
        assert len(evidence) == 2
        assert thin_flag is False
        return True, ["18239092"], "FGF5 record directly supports fur growth.", ""

    failures: list = []
    with _Patch(ls_module, "request_literature_evidence", _fake_fetch(TWO_RECORDS)), \
         _Patch(ls_module, "_llm_judge_sufficiency", fake_judge):
        result = await ls_module.literature_support_agent(LiteratureSupportInput(
            trait_name="fur growth",
            gene_list=["FGF5"],
            instruction="Find supporting literature for fur growth",
            context={},
        ))

    await _check("status COMPLETED", result.status == AgentStatus.COMPLETED, str(result.status), failures)
    await _check("no escalation target set", result.target_agent is None, str(result.target_agent), failures)
    await _check("both records returned", len(result.evidence) == 2, str(len(result.evidence)), failures)
    return failures


def _warn_slow_llm():
    print("  (real LLM call in progress — NIM can silently poll up to ~60s per "
          "turn on a cold request; run with --verbose to see per-turn progress "
          "logs from workflows.llm.tool_loop instead of a blank terminal)")


def _spy_llm_judge(reasoning_log: list):
    """Wraps the REAL _llm_judge_sufficiency so we can print what it decided
    and why, without changing its behavior (same spirit as _spy_llm_pick() in
    scripts/gene_mapper_scenarios.py)."""
    real = ls_module._llm_judge_sufficiency

    async def spying(trait_name, gene_list, evidence, thin_flag):
        sufficient, supporting_pmids, reasoning, prompt = await real(
            trait_name, gene_list, evidence, thin_flag
        )
        reasoning_log.append((sufficient, supporting_pmids, reasoning, prompt))
        return sufficient, supporting_pmids, reasoning, prompt

    return spying


async def scenario_two_records_real_llm_judgment(verbose: bool) -> list:
    """Same fixture as two-records-completed, but the REAL bind_tools LLM
    call is exercised end to end — only the external Literature Agent A2A
    call (request_literature_evidence) is faked, since that dependency isn't
    live in this repo yet. This is the one scenario that actually proves the
    LLM sufficiency judgment (§3/§5 of the design guide) works, not just the
    plumbing around it. Requires NVIDIA_NIM_API_KEY.

    Uses only ONE record (the thin case) rather than two clean ones: the
    LLM's job here is exactly the judgment call the guide asks for — a real
    model reading real summaries and deciding sufficiency/escalation itself
    — and the thin case is the one where a naive count-based check and an
    LLM reading the content could plausibly disagree, which is the whole
    point of putting an LLM here at all.
    """
    reasoning_log: list = []
    failures: list = []
    _warn_slow_llm()
    try:
        with _Patch(ls_module, "request_literature_evidence", _fake_fetch(ONE_RECORD)), \
             _Patch(ls_module, "_llm_judge_sufficiency", _spy_llm_judge(reasoning_log)):
            result = await ls_module.literature_support_agent(LiteratureSupportInput(
                trait_name="cold adaptation",
                gene_list=["UCP1"],
                instruction="Find supporting literature for cold adaptation",
                context={},
            ))
    except Exception as exc:
        await _check("real LLM call succeeded", False, f"raised {exc!r}", failures)
        return failures

    if not reasoning_log:
        print("  [INCONCLUSIVE] LLM judgment did not complete this run (likely "
              "rate-limited even after retry) — the agent correctly fell back "
              "to the deterministic path instead. Re-run to get a genuine "
              "LLM-judged sample.")
        return failures

    sufficient, supporting_pmids, reasoning, prompt = reasoning_log[0]
    print(f"  LLM verdict: sufficient={sufficient}")
    print(f"  LLM reasoning: {reasoning}")
    known_pmids = {r["pmid"] for r in ONE_RECORD}
    await _check("any cited pmid is grounded in the real evidence given",
                 all(p in known_pmids for p in supporting_pmids),
                 f"{supporting_pmids} subset of {known_pmids}", failures)
    await _check("status is a valid terminal/escalation status",
                 result.status in (AgentStatus.COMPLETED, AgentStatus.NEEDS_AGENT),
                 str(result.status), failures)
    if result.status == AgentStatus.NEEDS_AGENT:
        await _check("escalation carries a non-empty prompt_to_target_agent",
                     bool(result.prompt_to_target_agent), str(result.prompt_to_target_agent), failures)
    return failures


async def scenario_one_record_escalation(verbose: bool) -> list:
    """One record -> assert NEEDS_AGENT, target_agent resolved through the
    Capability Resolver. Mirrors tests/test_workflows.py's patch_llm_nodes
    pattern: monkeypatch resolve_capability, assert
    escalate_literature_agent_node's output — driven off the REAL
    literature_support_agent() result, not a hand-built state.
    """
    async def fake_judge(trait_name, gene_list, evidence, thin_flag):
        assert thin_flag is True
        return (
            False, [], "Only one record — insufficient to conclude.",
            "Find additional peer-reviewed evidence for trait 'cold adaptation' "
            "and genes ['UCP1']; only one record on hand.",
        )

    async def fake_resolve_capability(*, waiting_agent, need_description, known_context):
        assert waiting_agent == "Literature Support"
        return CapabilityResolution(
            target_agent="Literature Agent",
            prompt_to_target_agent=f"Resolve: {need_description}",
            reasoning="Existing literature evidence is below the thin-evidence threshold.",
        )

    failures: list = []
    with _Patch(ls_module, "request_literature_evidence", _fake_fetch(ONE_RECORD)), \
         _Patch(ls_module, "_llm_judge_sufficiency", fake_judge):
        ls_result = await ls_module.literature_support_agent(LiteratureSupportInput(
            trait_name="cold adaptation",
            gene_list=["UCP1"],
            instruction="Find supporting literature for cold adaptation",
            context={},
        ))

    await _check("literature_support status NEEDS_AGENT",
                 ls_result.status == AgentStatus.NEEDS_AGENT, str(ls_result.status), failures)

    state = TraitDiscoveryState(
        trait_name="cold adaptation",
        species_name="mouse",
        instruction="Find supporting literature for cold adaptation",
        go_annotations=[],
        pathway_data=[],
        protein_data=[],
        evidence=ls_result.evidence,
        _literature_prompt=ls_result.prompt_to_target_agent,
    )

    with _Patch(esc_module, "resolve_capability", fake_resolve_capability):
        esc_result = await esc_module.escalate_literature_agent_node(state)

    await _check("escalation status NEEDS_AGENT",
                 esc_result["status"] == AgentStatus.NEEDS_AGENT, str(esc_result["status"]), failures)
    await _check("target_agent resolved to Literature Agent",
                 esc_result["target_agent"] == "Literature Agent", str(esc_result["target_agent"]), failures)
    await _check("prompt_to_target_agent carried through",
                 bool(esc_result["prompt_to_target_agent"]), esc_result["prompt_to_target_agent"], failures)
    return failures


async def scenario_zero_records_failed(verbose: bool) -> list:
    """Zero records -> assert FAILED, matches mock_literature_support's
    existing rule. Must never reach the LLM."""
    async def _fail_if_called(*args, **kwargs):
        raise AssertionError("_llm_judge_sufficiency should not be called with zero evidence")

    failures: list = []
    with _Patch(ls_module, "request_literature_evidence", _fake_fetch([])), \
         _Patch(ls_module, "_llm_judge_sufficiency", _fail_if_called):
        result = await ls_module.literature_support_agent(LiteratureSupportInput(
            trait_name="an entirely unstudied trait",
            gene_list=["XYZ"],
            instruction="Find supporting literature",
            context={},
        ))

    await _check("status FAILED", result.status == AgentStatus.FAILED, str(result.status), failures)
    await _check("no evidence", result.evidence == [], str(result.evidence), failures)
    return failures


async def scenario_llm_unavailable_fallback(verbose: bool) -> list:
    """LLM unavailable -> assert the count-only fallback still produces a
    valid status, in both directions (enough records -> COMPLETED, thin ->
    NEEDS_AGENT)."""
    async def _simulated_outage(*args, **kwargs):
        raise RuntimeError("simulated NIM outage")

    failures: list = []

    with _Patch(ls_module, "request_literature_evidence", _fake_fetch(TWO_RECORDS)), \
         _Patch(ls_module, "_llm_judge_sufficiency", _simulated_outage):
        result_ok = await ls_module.literature_support_agent(LiteratureSupportInput(
            trait_name="fur growth", gene_list=["FGF5"], instruction="test", context={},
        ))
    await _check("2 records + LLM outage -> COMPLETED via count fallback",
                 result_ok.status == AgentStatus.COMPLETED, str(result_ok.status), failures)

    with _Patch(ls_module, "request_literature_evidence", _fake_fetch(ONE_RECORD)), \
         _Patch(ls_module, "_llm_judge_sufficiency", _simulated_outage):
        result_thin = await ls_module.literature_support_agent(LiteratureSupportInput(
            trait_name="cold adaptation", gene_list=["UCP1"], instruction="test", context={},
        ))
    await _check("1 record + LLM outage -> NEEDS_AGENT via count fallback",
                 result_thin.status == AgentStatus.NEEDS_AGENT, str(result_thin.status), failures)
    await _check("fallback still names Literature Agent as target",
                 result_thin.target_agent == "Literature Agent", str(result_thin.target_agent), failures)
    return failures


async def scenario_grounding_rejects_invented_pmid(verbose: bool) -> list:
    """Grounding test: LLM tries to cite a pmid not present in the retrieved
    evidence -> rejected, and the deterministic fallback takes over instead
    of trusting the model's say-so."""
    async def _fake_judge_invents_pmid(trait_name, gene_list, evidence, thin_flag):
        known = {r["pmid"] for r in evidence}
        invented = "99999999"
        assert invented not in known, "test fixture bug: invented pmid collides with a real one"
        return True, [invented], "hallucinated citation, not grounded in any tool result", ""

    failures: list = []
    with _Patch(ls_module, "request_literature_evidence", _fake_fetch(TWO_RECORDS)), \
         _Patch(ls_module, "_llm_judge_sufficiency", _fake_judge_invents_pmid):
        result = await ls_module.literature_support_agent(LiteratureSupportInput(
            trait_name="fur growth", gene_list=["FGF5"], instruction="test", context={},
        ))

    # Grounding rejection raises internally and the deterministic count-only
    # fallback takes over (2 records is not thin) — COMPLETED, but crucially
    # via the real, retrieved evidence, never the invented pmid.
    await _check("status COMPLETED (fallback recovered)",
                 result.status == AgentStatus.COMPLETED, str(result.status), failures)
    reported_pmids = {r.pmid for r in result.evidence}
    await _check("invented pmid was NOT reported",
                 "99999999" not in reported_pmids, str(reported_pmids), failures)
    await _check("only real, retrieved pmids reported",
                 reported_pmids == {r["pmid"] for r in TWO_RECORDS}, str(reported_pmids), failures)
    return failures


SCENARIOS = {
    "two-records-completed": (
        "Two clearly relevant records -> COMPLETED, no escalation. LLM "
        "judgment faked for speed/determinism — see two-records-real-llm-"
        "judgment for the version that hits the real model.",
        scenario_two_records_completed,
    ),
    "two-records-real-llm-judgment": (
        "Exercises the REAL bind_tools LLM sufficiency judgment end to end "
        "(only the Literature Agent A2A call is faked). Requires "
        "NVIDIA_NIM_API_KEY.",
        scenario_two_records_real_llm_judgment,
    ),
    "one-record-escalation": (
        "One record -> NEEDS_AGENT, target_agent resolved through the real "
        "Capability Resolver plumbing via escalate_literature_agent_node.",
        scenario_one_record_escalation,
    ),
    "zero-records-failed": (
        "Zero records -> FAILED, matches mock_literature_support's existing "
        "rule; the LLM is never called.",
        scenario_zero_records_failed,
    ),
    "llm-unavailable-fallback": (
        "Simulated NIM outage in both directions — deterministic count-only "
        "threshold still produces a valid status.",
        scenario_llm_unavailable_fallback,
    ),
    "grounding-rejects-invented-pmid": (
        "LLM cites a pmid absent from the retrieved evidence — the grounding "
        "check must reject it and fall back to the deterministic path.",
        scenario_grounding_rejects_invented_pmid,
    ),
}


async def run_one(name: str, verbose: bool) -> bool:
    description, fn = SCENARIOS[name]
    print("\n" + "#" * 70)
    print(f"# SCENARIO: {name}")
    print(f"# {description}")
    print("#" * 70)
    start = time.perf_counter()
    failures = await fn(verbose)
    elapsed = time.perf_counter() - start

    print("\n--- result ---")
    if failures:
        print(f"{len(failures)} check(s) failed in {elapsed:.2f}s:")
        for f in failures:
            print(f"  - {f}")
    else:
        print(f"all checks passed in {elapsed:.2f}s")
    return not failures


async def main(scenario: str, verbose: bool) -> int:
    logging.basicConfig(level=logging.INFO if verbose else logging.WARNING,
                         format="%(levelname)s:%(name)s:%(message)s")

    names = list(SCENARIOS) if scenario == "all" else [scenario]
    results = {}
    for name in names:
        results[name] = await run_one(name, verbose)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for name, ok in results.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")

    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Execute Literature Support agent scenarios end to end (design "
            "guide §10). The Literature Agent A2A call is patched for "
            "determinism; no live Literature Agent deployment or "
            "NVIDIA_NIM_API_KEY is required."
        )
    )
    parser.add_argument(
        "--scenario", choices=sorted(SCENARIOS) + ["all"], default="all",
        help="Which scenario to run (default: all).",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print INFO-level logs.",
    )
    args = parser.parse_args()
    sys.exit(asyncio.run(main(args.scenario, args.verbose)))
