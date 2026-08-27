"""
Reconstruction-handoff verification script
============================================

Answers one question end to end, with nothing hand-waved: *if the Genome
Agent decides an assembly needs reconstruction, is the exact payload it
would send to the Reconstruction Agent well-formed, complete, and JSON-safe?*

This is deliberately NOT another pytest file. `tests/test_reconstruction_path.py`
and `tests/test_gap_finder.py` already unit-test the pieces in isolation. This
script instead:

  1. Fakes NCBI at the lowest possible boundary — `ncbi_get()` in
     `subagents/_ncbi_client.py` — rather than mocking `find_target_gaps()`
     itself. That means the REAL code in `subagents/gap_finder.py` actually
     runs: the assembly→UID resolution, the accession-version lookup, the
     feature-table parser, and the real `sequence_window.fetch_sequence_window`
     calls for flanking sequence. Only the network call underneath all of
     that is faked, using response shapes copied from real NCBI eutils
     output. If the parsing regexes or the resolution chain were wrong,
     this script — not just a mocked unit test — would catch it.
  2. Drives the *real* compiled LangGraph orchestrator end to end (only
     species/metadata/annotation/LLM-capability-resolution are stubbed,
     since those are outside this feature's scope and already covered
     elsewhere).
  3. Takes the resulting `AgentResult` and reassembles the actual wire
     payload a caller would send to the Reconstruction Agent:
     `{"instruction": result.prompt_to_target_agent, "context": result.output}`
  4. Runs that payload through `validate_reconstruction_payload()`, a
     standalone checker written against the Reconstruction Agent's
     documented contract (docs/genome_agent_integration.md §9a) —
     independent of whatever the adapter happens to produce, so a
     regression in the adapter can't silently "validate" itself.
  5. Repeats for the degraded path (NCBI unreachable while gap-finding)
     and the negative/regression path (Chromosome-level assembly must
     NOT hand off at all).

Run:
    python -m genome_agent.scripts.run_reconstruction_handoff_verification
or:
    cd genome_agent && python scripts/run_reconstruction_handoff_verification.py

Exit code is 0 iff every scenario passes — safe to wire into CI.
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import sys
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from genome_agent.orchestrator import GenomeAgentLangGraphOrchestrator
from genome_agent.orchestrator_adapter import to_result
from genome_agent.schemas import AgentStatus

_WIDTH = 72


def _rule(title: str = "") -> None:
    print("=" * _WIDTH)
    if title:
        print(title)
        print("=" * _WIDTH)


# ===========================================================================
# Fake NCBI, at the ncbi_get() boundary
# ===========================================================================
#
# Scenario is the bear example from the actual integration ask:
#   scientific_name=Ursus maritimus, assembly_id=GCF_000687225.1,
#   sequence_accession=NW_007907101.1, assembly_level=Scaffold,
#   two gaps at (125430-125474) and (891203-891272).

ASSEMBLY_ID = "GCF_000687225.1"
SCIENTIFIC_NAME = "Ursus maritimus"
ASSEMBLY_LEVEL = "Scaffold"
ASSEMBLY_UID = "50000111"
NUCCORE_UID = "60000222"
SEQUENCE_ACCESSION = "NW_007907101.1"

_GAP_1 = {"start": 125430, "end": 125474, "length": 45}
_GAP_2 = {"start": 891203, "end": 891272, "length": 70}

# A real NCBI feature-table (.ft) response has 0-indented feature lines and
# tab-indented qualifier lines. Two `assembly_gap` features + one ordinary
# `gene` feature thrown in, to prove the parser only picks up gaps and
# correctly closes out the previous feature when a new one starts.
_FAKE_FEATURE_TABLE = f"""\
>Feature {SEQUENCE_ACCESSION}
1\t124000\tgene
\t\t\tgene\tLOC000001
{_GAP_1['start']}\t{_GAP_1['end']}\tassembly_gap
\t\t\testimated_length\t{_GAP_1['length']}
\t\t\tgap_type\twithin scaffold
{_GAP_2['start']}\t{_GAP_2['end']}\tassembly_gap
\t\t\testimated_length\t{_GAP_2['length']}
\t\t\tgap_type\twithin scaffold
900000\t950000\tgene
\t\t\tgene\tLOC000002
"""


def _fake_flank_fasta(seq_start: int, seq_stop: int) -> str:
    """Deterministic pseudo-sequence so left/right flanks are distinguishable
    and reproducible without a real download."""
    bases = "ACGT"
    body = "".join(bases[(seq_start + i) % 4] for i in range(max(0, seq_stop - seq_start + 1)))
    return f">{SEQUENCE_ACCESSION}:{seq_start}-{seq_stop}\n{body}\n"


class FakeResponse:
    def __init__(self, *, json_data: dict | None = None, text: str = ""):
        self._json_data = json_data
        self.text = text

    def json(self) -> dict:
        return self._json_data or {}

    def raise_for_status(self) -> None:
        return None


def _happy_path_ncbi_get(params: dict) -> FakeResponse:
    """Faithful stand-in for `ncbi_get`, keyed on the same params real code sends."""
    path = params.get("path")

    if path == "esearch.fcgi" and params.get("db") == "assembly":
        assert params["term"] == f"{ASSEMBLY_ID}[Assembly]", params["term"]
        return FakeResponse(json_data={"esearchresult": {"idlist": [ASSEMBLY_UID]}})

    if path == "elink.fcgi":
        assert params["id"] == ASSEMBLY_UID, params["id"]
        # GCF_ accessions try assembly_nuccore_refseq first (see sequence_window.py)
        if params.get("linkname") == "assembly_nuccore_refseq":
            return FakeResponse(
                json_data={
                    "linksets": [
                        {"linksetdbs": [{"links": [NUCCORE_UID]}]},
                    ]
                }
            )
        return FakeResponse(json_data={"linksets": []})

    if path == "esummary.fcgi":
        assert params["id"] == NUCCORE_UID, params["id"]
        return FakeResponse(
            json_data={
                "result": {
                    NUCCORE_UID: {
                        "accessionversion": SEQUENCE_ACCESSION,
                        "caption": SEQUENCE_ACCESSION.split(".")[0],
                    }
                }
            }
        )

    if path == "efetch.fcgi" and params.get("rettype") == "ft":
        assert params["id"] == NUCCORE_UID, params["id"]
        return FakeResponse(text=_FAKE_FEATURE_TABLE)

    if path == "efetch.fcgi" and params.get("rettype") == "fasta":
        assert params["id"] == NUCCORE_UID, params["id"]
        return FakeResponse(text=_fake_flank_fasta(params["seq_start"], params["seq_stop"]))

    raise AssertionError(f"Unexpected NCBI call in fake responder: {params}")


def _unreachable_ncbi_get(params: dict) -> FakeResponse:
    raise ConnectionError("simulated NCBI outage")


# ===========================================================================
# Orchestrator plumbing common to every scenario (species/metadata/annotation
# and the LLM capability resolver are outside this feature's scope, so they
# are stubbed the same way tests/test_reconstruction_path.py already does).
# ===========================================================================


def _make_decision(target_agent: str, handoff_message: str = "reconstruct this"):
    decision = MagicMock()
    decision.target_agent = target_agent
    decision.handoff_message = handoff_message
    return decision


def _metadata_for(level: str) -> dict:
    return {
        "genome_size_bp": 2_500_000_000,
        "assembly_level": level,
        "species_name": SCIENTIFIC_NAME,
    }


async def _run_orchestrator(*, assembly_level: str, ncbi_get_impl) -> Any:
    species = {"taxon_id": "29073", "scientific_name": SCIENTIFIC_NAME, "assembly_id": ASSEMBLY_ID}

    with (
        patch(
            "genome_agent.workflows.nodes.species_resolver_node.resolve_species",
            new=AsyncMock(return_value=species),
        ),
        patch(
            "genome_agent.workflows.nodes.species_resolver_node.resolve_species_llm",
            new=AsyncMock(return_value=None),  # forces the deterministic path above
        ),
        patch(
            "genome_agent.workflows.nodes.genome_data_nodes.get_genome_metadata",
            new=AsyncMock(return_value=_metadata_for(assembly_level)),
        ),
        patch(
            "genome_agent.workflows.nodes.genome_data_nodes.get_gene_annotation",
            new=AsyncMock(return_value={"gene_count": 0, "gene_list": []}),
        ),
        # The actual boundary under test: everything above find_target_gaps()
        # in subagents/gap_finder.py and subagents/sequence_window.py runs
        # for real against this fake transport.
        patch("genome_agent.subagents.gap_finder.ncbi_get", new=ncbi_get_impl),
        patch("genome_agent.subagents.sequence_window.ncbi_get", new=ncbi_get_impl),
        patch(
            "genome_agent.workflows.nodes.reconstruction_resolver_node.resolve_capability",
            return_value=_make_decision("Reconstruction Agent"),
        ),
        patch(
            "genome_agent.workflows.nodes.reconstruction_resolver_node.resolve_capability_fallback",
            return_value=_make_decision("Reconstruction Agent"),
        ),
        patch(
            "genome_agent.workflows.nodes.explanation_writer_node.write_explanation",
            new=AsyncMock(return_value="Assembly is incomplete; handing off."),
        ),
    ):
        orch = GenomeAgentLangGraphOrchestrator()
        state = await orch.run(
            user_question=f"Reconstruct the {SCIENTIFIC_NAME} genome.",
            species_name=SCIENTIFIC_NAME,
            visualization_scope="none",
        )
    return to_result(state)


# ===========================================================================
# The payload validator — written against the documented contract, not
# against whatever the adapter currently emits. This is the thing that
# would actually catch a regression.
# ===========================================================================

_ASSEMBLY_ACCESSION_RE = re.compile(r"^GC[AF]_\d+\.\d+$")
_SEQUENCE_ACCESSION_RE = re.compile(r"^[A-Z]{1,4}_?\d+(\.\d+)?$")
_ALLOWED_BASES_RE = re.compile(r"^[ACGTNacgtn]*$")


class PayloadValidationError(AssertionError):
    pass


def validate_reconstruction_payload(payload: dict, *, strict: bool) -> list[str]:
    """Check `payload` against the Reconstruction Agent's documented
    contract. Raises PayloadValidationError on any hard failure; returns a
    list of soft warnings (non-fatal but worth a human's attention) for
    conditions that are only violated in degraded (non-strict) scenarios.

    `strict=True`  — the happy path. Every field must be fully populated.
    `strict=False` — the degraded/NCBI-unreachable path. sequence_accession
        may be None and target_gaps may be empty; everything else still
        must hold.
    """
    warnings: list[str] = []

    def fail(msg: str) -> None:
        raise PayloadValidationError(msg)

    # -- JSON-safety: this is what actually crosses the wire --
    try:
        round_tripped = json.loads(json.dumps(payload))
    except (TypeError, ValueError) as exc:
        fail(f"payload is not JSON-serialisable: {exc}")
    if round_tripped != payload:
        fail("payload changed shape across a JSON round-trip")

    # -- top level --
    if set(payload.keys()) - {"instruction", "context"} and "instruction" not in payload:
        fail(f"unexpected top-level shape: {list(payload.keys())}")
    if "instruction" not in payload or "context" not in payload:
        fail(f"payload must have 'instruction' and 'context' keys, got {list(payload.keys())}")

    instruction = payload["instruction"]
    if not isinstance(instruction, str) or not instruction.strip():
        fail(f"instruction must be a non-empty string, got {instruction!r}")

    context = payload["context"]
    if not isinstance(context, dict):
        fail(f"context must be a dict, got {type(context)}")

    required_keys = {
        "scientific_name",
        "assembly_id",
        "sequence_accession",
        "assembly_level",
        "target_gaps",
    }
    missing = required_keys - context.keys()
    if missing:
        fail(f"context is missing required keys: {sorted(missing)}")

    # -- scientific_name --
    if not isinstance(context["scientific_name"], str) or not context["scientific_name"].strip():
        fail(f"scientific_name must be a non-empty string, got {context['scientific_name']!r}")

    # -- assembly_id --
    assembly_id = context["assembly_id"]
    if not isinstance(assembly_id, str) or not _ASSEMBLY_ACCESSION_RE.match(assembly_id):
        fail(f"assembly_id doesn't look like a GCA_/GCF_ accession: {assembly_id!r}")

    # -- assembly_level --
    if context["assembly_level"] not in {"Scaffold", "Contig"}:
        fail(
            "assembly_level should be 'Scaffold' or 'Contig' on a reconstruction "
            f"handoff, got {context['assembly_level']!r}"
        )

    # -- sequence_accession --
    sequence_accession = context["sequence_accession"]
    if sequence_accession is None:
        if strict:
            fail("sequence_accession is None on the happy path — gap-finder resolution failed silently")
        else:
            warnings.append(
                "sequence_accession is None (expected in the degraded/NCBI-down scenario, "
                "but the Reconstruction Agent needs to explicitly tolerate this)"
            )
    elif not isinstance(sequence_accession, str) or not _SEQUENCE_ACCESSION_RE.match(sequence_accession):
        fail(f"sequence_accession doesn't look like an NCBI Nuccore accession: {sequence_accession!r}")

    # -- target_gaps --
    target_gaps = context["target_gaps"]
    if not isinstance(target_gaps, list):
        fail(f"target_gaps must be a list, got {type(target_gaps)}")
    if strict and not target_gaps:
        fail("target_gaps is empty on the happy path — no gaps were found/enriched")
    if len(target_gaps) > 5:
        warnings.append(f"target_gaps has {len(target_gaps)} entries — DEFAULT_MAX_GAPS=5 should cap this")

    for i, gap in enumerate(target_gaps):
        prefix = f"target_gaps[{i}]"
        for field in ("start", "end", "length", "left_flank", "right_flank"):
            if field not in gap:
                fail(f"{prefix} is missing '{field}'")

        for field in ("start", "end", "length"):
            if not isinstance(gap[field], int) or isinstance(gap[field], bool):
                fail(f"{prefix}.{field} must be an int, got {gap[field]!r}")

        if gap["start"] <= 0 or gap["end"] <= 0:
            fail(f"{prefix} has non-positive coordinates: start={gap['start']} end={gap['end']}")
        if gap["end"] < gap["start"]:
            fail(f"{prefix} has end < start: {gap['start']}..{gap['end']}")
        if gap["length"] <= 0:
            fail(f"{prefix}.length must be positive, got {gap['length']}")

        span = gap["end"] - gap["start"] + 1
        if gap["length"] != span:
            # Not a hard failure — gap_finder.py deliberately prefers
            # `estimated_length` over the raw coordinate span (see its
            # docstring) — but worth flagging if it ever silently diverges
            # by an implausible amount.
            warnings.append(
                f"{prefix}.length ({gap['length']}) != end-start+1 ({span}) — "
                "expected when estimated_length overrides the coordinate span, "
                "but double check this wasn't a parsing bug"
            )

        for flank_field in ("left_flank", "right_flank"):
            flank = gap[flank_field]
            if not isinstance(flank, str):
                fail(f"{prefix}.{flank_field} must be a string, got {type(flank)}")
            if strict and not flank:
                fail(f"{prefix}.{flank_field} is empty on the happy path")
            if flank and not _ALLOWED_BASES_RE.match(flank):
                fail(f"{prefix}.{flank_field} contains non-ACGTN characters: {flank!r}")

    return warnings


# ===========================================================================
# Scenarios
# ===========================================================================


def _to_wire_payload(result) -> dict:
    """This is the actual translation a caller performs: AgentResult's
    (target_agent, prompt_to_target_agent, output) triple becomes the
    {instruction, context} JSON body sent to the Reconstruction Agent."""
    return {"instruction": result.prompt_to_target_agent, "context": result.output}


async def scenario_happy_path() -> bool:
    _rule("SCENARIO A — Scaffold assembly, NCBI healthy (real bear example)")
    result = await _run_orchestrator(assembly_level=ASSEMBLY_LEVEL, ncbi_get_impl=_happy_path_ncbi_get)

    ok = True
    ok &= _check(result.status == AgentStatus.NEEDS_AGENT, f"status == NEEDS_AGENT (got {result.status})")
    ok &= _check(
        result.target_agent == "Reconstruction Agent",
        f"target_agent == 'Reconstruction Agent' (got {result.target_agent!r})",
    )

    payload = _to_wire_payload(result)
    print("\nWire payload that would be sent to the Reconstruction Agent:")
    print(json.dumps(payload, indent=2))

    try:
        warnings = validate_reconstruction_payload(payload, strict=True)
    except PayloadValidationError as exc:
        _check(False, f"payload passes strict validation ({exc})")
        return False
    ok &= _check(True, "payload passes strict validation against the documented contract")
    for w in warnings:
        print(f"  [warn] {w}")

    context = payload["context"]
    ok &= _check(context["scientific_name"] == SCIENTIFIC_NAME, "scientific_name matches species resolver output")
    ok &= _check(context["assembly_id"] == ASSEMBLY_ID, "assembly_id matches species resolver output")
    ok &= _check(
        context["sequence_accession"] == SEQUENCE_ACCESSION,
        f"sequence_accession is the resolved NW_ accession (got {context['sequence_accession']!r})",
    )
    ok &= _check(len(context["target_gaps"]) == 2, f"both feature-table gaps came through (got {len(context['target_gaps'])})")

    gaps_by_start = {g["start"]: g for g in context["target_gaps"]}
    for expected in (_GAP_1, _GAP_2):
        gap = gaps_by_start.get(expected["start"])
        ok &= _check(gap is not None, f"gap at start={expected['start']} is present")
        if gap:
            ok &= _check(gap["end"] == expected["end"], f"gap[{expected['start']}].end correct")
            ok &= _check(
                gap["length"] == expected["length"],
                f"gap[{expected['start']}].length correct (estimated_length qualifier honored)",
            )
            ok &= _check(len(gap["left_flank"]) > 0, f"gap[{expected['start']}].left_flank populated")
            ok &= _check(len(gap["right_flank"]) > 0, f"gap[{expected['start']}].right_flank populated")
    return ok


async def scenario_ncbi_unreachable() -> bool:
    _rule("SCENARIO B — Scaffold assembly, NCBI unreachable during gap-finding (degraded path)")
    result = await _run_orchestrator(assembly_level=ASSEMBLY_LEVEL, ncbi_get_impl=_unreachable_ncbi_get)

    ok = True
    ok &= _check(
        result.status == AgentStatus.NEEDS_AGENT,
        "handoff still proceeds (escalation is not dropped just because gap-finding failed)",
    )
    payload = _to_wire_payload(result)
    print("\nWire payload for the degraded path:")
    print(json.dumps(payload, indent=2))

    try:
        warnings = validate_reconstruction_payload(payload, strict=False)
    except PayloadValidationError as exc:
        _check(False, f"payload passes non-strict validation ({exc})")
        return False
    ok &= _check(True, "payload is still well-formed (sequence_accession=None, target_gaps=[])")
    for w in warnings:
        print(f"  [warn] {w}")

    context = payload["context"]
    ok &= _check(context["sequence_accession"] is None, "sequence_accession is None, not a stale/wrong value")
    ok &= _check(context["target_gaps"] == [], "target_gaps degrades to an empty list, not omitted")
    ok &= _check("warnings" in context, "the NCBI failure is surfaced as a 'warnings' key in context")
    if "warnings" in context:
        ok &= _check(
            any("find_target_gaps" in w for w in context["warnings"]),
            "the warning text identifies find_target_gaps as the failure point",
        )
    return ok


async def scenario_no_false_positive() -> bool:
    _rule("SCENARIO C — Chromosome-level assembly (regression check: must NOT hand off)")
    result = await _run_orchestrator(assembly_level="Chromosome", ncbi_get_impl=_happy_path_ncbi_get)

    ok = True
    ok &= _check(
        result.status == AgentStatus.COMPLETED,
        f"status == COMPLETED, no reconstruction handoff (got {result.status})",
    )
    ok &= _check(result.target_agent is None, "target_agent is None")
    ok &= _check(
        isinstance(result.output, dict) and "target_gaps" not in result.output,
        "generic output dict has no target_gaps/sequence_accession leakage",
    )
    return ok


# ===========================================================================
# Tiny check helper + main
# ===========================================================================

_checks_run = 0
_checks_failed = 0


def _check(condition: bool, description: str) -> bool:
    global _checks_run, _checks_failed
    _checks_run += 1
    mark = "✅" if condition else "❌"
    print(f"  {mark} {description}")
    if not condition:
        _checks_failed += 1
    return condition


async def main() -> int:
    results = {
        "happy_path": await scenario_happy_path(),
        "ncbi_unreachable": await scenario_ncbi_unreachable(),
        "no_false_positive": await scenario_no_false_positive(),
    }

    _rule("SUMMARY")
    for name, passed in results.items():
        print(f"  {'✅' if passed else '❌'} {name}")
    print(f"\n{_checks_run - _checks_failed}/{_checks_run} individual checks passed.")

    all_passed = all(results.values()) and _checks_failed == 0
    print("\n" + ("ALL SCENARIOS PASSED — payload and handoff wiring verified." if all_passed
                   else "SOME SCENARIOS FAILED — see ❌ marks above."))
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
