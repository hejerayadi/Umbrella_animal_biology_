"""Score the agent against bases it was never shown, through the real services.

Every test in `tests/` mocks the network edge. That proves the code runs; it
cannot tell a working evidence path from a plausible-looking one. Only this can:
a known region is withheld, the agent reconstructs it from homology alone, and
the answer is compared with what was actually there.

The battery is ordered by how hard the biology is, not by how hard the code is.
An easy case has many closely related sequenced genomes; a hard one has few, or
sits in a clade whose vernacular name is paraphyletic and resolves to nothing,
or is long enough that the search cannot finish inside the run deadline.

Run:  uv run python scripts/ground_truth.py
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from dataclasses import dataclass, field
from typing import Any

from fastapi.testclient import TestClient

from reconstruction_agent.config.settings import get_settings
from reconstruction_agent.integrations.ncbi.client import NcbiClient
from reconstruction_agent.main import create_app


@dataclass(frozen=True, slots=True)
class Case:
    """One withheld region, and why it is interesting."""

    name: str
    difficulty: str
    accession: str
    start: int
    length: int
    rationale: str


@dataclass
class Outcome:
    """What the agent did with one case."""

    case: Case
    status: str = ""
    gap_status: str = ""
    exact: bool = False
    identity: float | None = None
    confidence: float | None = None
    level: str = ""
    database: str | None = None
    spanning_hits: int = 0
    unresolved_reason: str | None = None
    seconds: float = 0.0
    note: str = ""
    scores: dict[str, Any] = field(default_factory=dict)


#: Ordered easiest to hardest. Every one is a real record with a real answer.
CASES: tuple[Case, ...] = (
    Case(
        name="human mitogenome",
        difficulty="1 trivial",
        accession="NC_012920.1",
        start=5000,
        length=45,
        rationale="The most sequenced molecule in biology; thousands of near-identical relatives.",
    ),
    Case(
        name="polar bear mitogenome",
        difficulty="2 easy",
        accession="NC_003428.1",
        start=5000,
        length=45,
        rationale="Many conspecific mitogenomes. The case the design was built from.",
    ),
    Case(
        name="polar bear, 200-base gap",
        difficulty="3 moderate",
        accession="NC_003428.1",
        start=7000,
        length=200,
        rationale="Same evidence, a much longer hole: tests whether length degrades the fill.",
    ),
    Case(
        name="chicken mitogenome",
        difficulty="4 moderate",
        accession="NC_001323.1",
        start=5000,
        length=60,
        rationale="A bird, not a mammal. Tests that clade selection generalises.",
    ),
    Case(
        name="zebrafish mitogenome",
        difficulty="5 harder",
        accession="NC_002333.2",
        start=5000,
        length=60,
        rationale="Falls in the vertebrate division that excludes mammals - the original trap.",
    ),
    Case(
        name="honeybee mitogenome",
        difficulty="6 hard",
        accession="NC_001566.1",
        start=5000,
        length=60,
        rationale="Invertebrate is paraphyletic and resolves to no taxon; tests the fallback.",
    ),
    Case(
        name="polar bear nuclear scaffold",
        difficulty="7 very hard",
        accession="NW_007907101.1",
        start=0,
        length=0,
        rationale="Nuclear search measured at ~550 s against a 300 s deadline. Should abstain.",
    ),
)


async def withheld(accession: str, start: int, length: int) -> str:
    """The bases the agent will not be shown."""
    ncbi = NcbiClient(get_settings().ncbi)
    try:
        record = await ncbi.fetch_sequence(accession)
        return record.residues[start : start + length]
    finally:
        await ncbi.aclose()


def run_case(client: TestClient, case: Case) -> Outcome:
    """Withhold a region, ask for it back, and score the answer."""
    outcome = Outcome(case=case)

    context: dict[str, Any] = {"sequence_accession": case.accession}
    expected = ""
    if case.length > 0:
        try:
            expected = asyncio.run(withheld(case.accession, case.start, case.length))
        except Exception as error:  # noqa: BLE001 - a harness failure, not an agent one
            outcome.note = f"could not fetch reference bases: {error}"
            return outcome
        if len(expected) != case.length:
            outcome.note = "record too short for the requested region"
            return outcome
        context["target_gaps"] = [{"start": case.start, "end": case.start + case.length}]

    started = time.monotonic()
    response = client.post(
        "/execute",
        json={"instruction": "Reconstruct the unresolved regions.", "context": context},
        headers={"X-Trace-Id": f"gt-{case.accession}"},
    )
    outcome.seconds = time.monotonic() - started
    outcome.status = response.json().get("status", "?")

    payload = response.json().get("output")
    if not isinstance(payload, dict):
        outcome.note = str(payload)[:200]
        return outcome

    gaps = payload["reconstruction"]["reconstructions"]
    if not gaps:
        outcome.note = "no gaps reported"
        return outcome

    gap = gaps[0]
    outcome.gap_status = gap["status"]
    outcome.unresolved_reason = gap.get("unresolved_reason")
    outcome.spanning_hits = gap["evidence"]["homology"]["gap_spanning_hits"]
    searched = gap["provenance"]["databases_searched"]
    outcome.database = searched[-1] if searched else None

    candidate = gap.get("selected_candidate")
    if candidate:
        got = candidate["sequence"]
        outcome.confidence = candidate["confidence"]
        outcome.level = candidate["confidence_level"]
        outcome.scores = candidate["scores"]
        outcome.exact = got == expected
        if expected and len(got) == len(expected):
            same = sum(a == b for a, b in zip(got, expected, strict=True))
            outcome.identity = same / len(expected)
        elif expected:
            outcome.note = f"length {len(got)} vs expected {len(expected)}"
    return outcome


def report(outcomes: list[Outcome]) -> None:
    """Print the result table. No rounding in the agent favour."""
    print()
    print("=" * 108)
    print("GROUND TRUTH - reconstruction scored against withheld bases")
    print("=" * 108)
    header = (
        f"{'difficulty':<13} {'case':<28} {'gap':<11} {'exact':<6} "
        f"{'ident':<7} {'conf':<6} {'span':<5} {'sec':<5} database"
    )
    print(header)
    print("-" * 108)

    for item in outcomes:
        identity = f"{item.identity:.3f}" if item.identity is not None else "-"
        confidence = f"{item.confidence:.2f}" if item.confidence is not None else "-"
        exact = "YES" if item.exact else ("no" if item.gap_status == "RESOLVED" else "-")
        print(
            f"{item.case.difficulty:<13} {item.case.name:<28} "
            f"{item.gap_status or item.status:<11} {exact:<6} "
            f"{identity:<7} {confidence:<6} {item.spanning_hits:<5} "
            f"{item.seconds:<5.0f} {item.database or '-'}"
        )
        if item.unresolved_reason:
            print(f"{'':<13} reason: {item.unresolved_reason}")
        if item.note:
            print(f"{'':<13} note: {item.note}")

    print("-" * 108)
    resolved = [o for o in outcomes if o.gap_status == "RESOLVED"]
    exact = [o for o in resolved if o.exact]
    print(
        f"attempted {len(outcomes)} | resolved {len(resolved)} | "
        f"exact {len(exact)} | abstained {len(outcomes) - len(resolved)}"
    )
    print(json.dumps([_row(o) for o in outcomes], indent=2))


def _row(outcome: Outcome) -> dict[str, Any]:
    return {
        "case": outcome.case.name,
        "difficulty": outcome.case.difficulty,
        "accession": outcome.case.accession,
        "rationale": outcome.case.rationale,
        "gap_status": outcome.gap_status or outcome.status,
        "exact": outcome.exact,
        "identity": outcome.identity,
        "confidence": outcome.confidence,
        "confidence_level": outcome.level,
        "database": outcome.database,
        "gap_spanning_hits": outcome.spanning_hits,
        "unresolved_reason": outcome.unresolved_reason,
        "seconds": round(outcome.seconds, 1),
        "scores": outcome.scores,
        "note": outcome.note,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", help="substring of a case name to run just one")
    args = parser.parse_args()

    cases = [c for c in CASES if not args.only or args.only.lower() in c.name.lower()]

    # Entered as a context manager on purpose. Outside one, TestClient starts
    # and stops an event loop per request, while the pooled HTTP clients are
    # cached on app.state at the first request and stay bound to the first
    # loop - so the second case dies with "Event loop is closed". Under uvicorn
    # there is one loop for the process lifetime, which is the condition the
    # shared clients are built for, and this reproduces it.
    outcomes: list[Outcome] = []
    with TestClient(create_app(), raise_server_exceptions=False) as client:
        for case in cases:
            print(f"--- {case.difficulty}: {case.name} ({case.accession}) ---", flush=True)
            outcomes.append(run_case(client, case))
            print(f"    done in {outcomes[-1].seconds:.0f}s", flush=True)

    report(outcomes)
    return 0


if __name__ == "__main__":
    sys.exit(main())
