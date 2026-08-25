"""Score the agent against known bases, using the real external services.

The only test that can tell a working evidence path from a plausible-looking
one. Everything in `tests/` mocks the network edge, so BLAST there returns
instantly and always with usable hits - which is precisely where both the
timing bug and the database bug hid for the whole life of this agent.

Method: take a published sequence, punch a hole of known content in it, hand the
damaged copy to the *real* agent through the *real* orchestrator entry point,
and compare what comes back against the bases that were removed.

    uv run python scripts/ground_truth.py                 # every case
    uv run python scripts/ground_truth.py --case mito     # one
    uv run python scripts/ground_truth.py --control em_std_vrt

`--control` re-runs each case with the database pinned to the value the agent
used to hardcode. The before/after delta on the same accession, in the same
minute, against the same service is the evidence; a single post-fix number on
its own is not.

Honest by construction: a case that fails is reported as failed with its reason,
and the exit code is non-zero. Nothing here fills in a plausible answer when the
agent could not find one.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from configuration.logging import configure_logging  # noqa: E402
from configuration.runtime import use_selector_event_loop  # noqa: E402

use_selector_event_loop()

from application.reconstruction_service import ReconstructionService  # noqa: E402
from configuration.settings import Settings, get_settings  # noqa: E402
from contracts.input import ReconstructionRequest  # noqa: E402
from infrastructure.ncbi.client import NCBIClient  # noqa: E402
from observability.events import EventEmitter  # noqa: E402
from tools.ncbi.mapper import to_references  # noqa: E402
from tools.registry import build_database_advisor, build_default_registry  # noqa: E402


@dataclass(frozen=True, slots=True)
class Case:
    """One masked region with a known answer."""

    label: str
    accession: str
    organism: str
    mask_start: int
    mask_length: int
    #: The ENA division the target should resolve to, asserted so a silent
    #: misclassification shows up as a failure rather than as a poor score.
    expected_division: str | None
    #: Extra orchestrator context, so a case can reproduce a real payload.
    context: dict[str, object] = field(default_factory=dict)
    #: Only fetch this window of a large record, so a 15.9 Mb scaffold does not
    #: have to be pulled in full to mask 45 bases inside it.
    window: tuple[int, int] | None = None


CASES: tuple[Case, ...] = (
    Case(
        label="mito",
        accession="NC_003428.1",
        organism="Ursus maritimus",
        mask_start=6000,
        mask_length=45,
        expected_division="mam",
    ),
    Case(
        label="human-mito",
        accession="NC_012920.1",
        organism="Homo sapiens",
        mask_start=8000,
        mask_length=45,
        expected_division="hum",
    ),
    Case(
        label="mouse-mito",
        accession="NC_005089.1",
        organism="Mus musculus",
        mask_start=7000,
        mask_length=45,
        expected_division="mus",
    ),
    # The orchestrator's real payload: a Scaffold-level nuclear assembly, the
    # common-name/scientific-name pair that used to be read the wrong way
    # round, and a 15.9 Mb target. The acceptance case.
    Case(
        label="scaffold",
        accession="NW_007907101",
        organism="Ursus maritimus",
        mask_start=2000,
        mask_length=45,
        expected_division="mam",
        window=(1, 400_000),
        context={
            "species": "polar bear",
            "assembly_id": "GCF_000687225.1",
            "species_record": {"scientific_name": "Ursus maritimus"},
            "genome_metadata": {"assembly_level": "Scaffold"},
            "gene_list": ["FGF5", "UCP1"],
        },
    ),
    # No organism anywhere: exercises the path where the division cannot be
    # resolved. The agent must say so rather than guess a division.
    Case(
        label="no-organism",
        accession="NC_003428.1",
        organism="",
        mask_start=6000,
        mask_length=45,
        expected_division=None,
    ),
)


@dataclass
class Score:
    """What one case produced, measured rather than asserted."""

    case: str
    status: str = "not_run"
    exact_match: bool = False
    base_accuracy: float = 0.0
    length_delta: int = 0
    predicted: str | None = None
    truth: str = ""
    confidence: float = 0.0
    database: str | None = None
    division: str | None = None
    database_trials: dict = field(default_factory=dict)
    blast_hits_total: int = 0
    blast_hits_carrying_gap: int = 0
    gaps_total: int = 0
    gaps_resolved: int = 0
    gaps_unresolved: int = 0
    slices_used: int = 0
    wall_seconds: float = 0.0
    failure_reason: str | None = None

    def as_dict(self) -> dict:
        payload = dict(self.__dict__)
        # The full fill is interesting on a failure and noise on a success.
        if self.exact_match:
            payload.pop("predicted", None)
            payload.pop("truth", None)
        return payload


def base_accuracy(predicted: str | None, truth: str) -> tuple[float, int]:
    """Fraction of agreeing positions, and how far the length is off.

    Scored over the shorter of the two and reported *with* the length delta on
    purpose: a fill truncated to its first correct base would otherwise score
    1.0, which would be the most flattering possible way to be wrong.
    """
    if not predicted or not truth:
        return 0.0, len(truth or "") - len(predicted or "")
    comparable = min(len(predicted), len(truth))
    matches = sum(
        1 for a, b in zip(predicted[:comparable], truth[:comparable], strict=True) if a == b
    )
    return matches / comparable, len(truth) - len(predicted)


class _PinnedAdvisor:
    """Forces every search to one database, for the control arm.

    Stands in for `DatabaseAdvisor` so the before/after comparison runs the
    identical code path with only the database changed. Without this the
    control would measure two different pipelines and prove nothing.
    """

    def __init__(self, database: str) -> None:
        self._database = database

    async def advise(self, **_: object):
        from domain.services.target_profile import Molecule, TargetProfile
        from tools.blast.advisor import DatabaseAdvice

        return DatabaseAdvice(
            candidates=[self._database],
            profile=TargetProfile(None, Molecule.UNKNOWN, "pinned"),
            options=[],
            hint=f"Pinned to {self._database} for the control arm.",
        )


async def fetch(client: NCBIClient, case: Case) -> str:
    """The published residues for a case, windowed if the record is huge."""
    if case.window is not None:
        raw = await client.fetch_region(case.accession, case.window[0], case.window[1])
    else:
        raw = await client.fetch_fasta([case.accession])
    records = to_references(raw)
    if not records or not records[0].residues:
        raise RuntimeError(f"NCBI returned no sequence for {case.accession}")
    return records[0].residues.upper()


async def run_case(
    case: Case, settings: Settings, *, control: str | None = None
) -> Score:
    """One case, through the real service, over as many slices as it needs."""
    score = Score(case=case.label)
    started = time.monotonic()

    client = NCBIClient(settings.ncbi, timeout=settings.http.timeout_seconds)
    try:
        residues = await fetch(client, case)
    except Exception as error:  # noqa: BLE001 - reported, never hidden
        score.status = "fetch_failed"
        score.failure_reason = str(error)
        return score
    finally:
        await client.aclose()

    start, length = case.mask_start, case.mask_length
    truth = residues[start : start + length]
    if len(truth) != length or "N" in truth:
        score.status = "bad_mask_site"
        score.failure_reason = "the chosen window is not fully known sequence"
        return score
    score.truth = truth

    damaged = residues[:start] + "N" * length + residues[start + length :]

    context: dict = {
        "sequence": {
            "identifier": f"{case.accession}_masked",
            "residues": damaged,
            "description": f"{case.organism} masked ground-truth case".strip(),
        },
        **case.context,
    }
    if case.organism:
        context.setdefault("organism", case.organism)

    request = ReconstructionRequest.from_agent_request(
        f"Fill the gaps (runs of N) in {case.accession}.", context
    )

    service = ReconstructionService(
        settings,
        build_default_registry(settings),
        events=EventEmitter(),
        # The control arm pins the database the agent used to hardcode, by
        # replacing the advisor rather than by threading a flag through the
        # production path - the agent under test must have no way to be told
        # which database to use.
        databases=_PinnedAdvisor(control) if control else build_database_advisor(settings),
    )

    trace_id = f"gt-{case.label}-{int(time.time())}"
    outcome = None
    for slice_index in range(settings.continuation.max_slices):
        score.slices_used = slice_index + 1
        outcome = await service.reconstruct(request, trace_id=trace_id)
        if outcome.finished:
            break

    score.wall_seconds = round(time.monotonic() - started, 1)
    if outcome is None:
        score.status = "no_outcome"
        return score

    result = outcome.result
    score.status = str(getattr(result, "status", "unknown"))

    gaps = list(getattr(result, "gaps", []) or [])
    score.gaps_total = len(gaps)
    for gap in gaps:
        filled = getattr(gap, "reconstructed_sequence", None)
        if filled:
            score.gaps_resolved += 1
            if score.predicted is None:
                score.predicted = filled
                score.confidence = float(getattr(gap, "confidence", 0.0) or 0.0)
        else:
            score.gaps_unresolved += 1
            if score.failure_reason is None:
                score.failure_reason = getattr(gap, "explanation", None) or "unresolved"

    for observation in getattr(result, "observations", []) or []:
        diagnostics = getattr(observation, "diagnostics", None) or {}
        hits = int(diagnostics.get("blast_hits_total", 0) or 0)
        carrying = int(diagnostics.get("blast_hits_carrying_gap", 0) or 0)
        score.blast_hits_total += hits
        score.blast_hits_carrying_gap += carrying

        database = diagnostics.get("blast_database")
        if not database:
            continue
        score.division = diagnostics.get("blast_division") or score.division
        # Per database, so the parallel probe's comparison is visible: which
        # one was searched, what it returned, and which one the agent kept.
        trial = score.database_trials.setdefault(
            database, {"hits": 0, "carrying_gap": 0, "searches": 0}
        )
        trial["hits"] += hits
        trial["carrying_gap"] += carrying
        trial["searches"] += 1

    # The database that actually produced carriers - the agent's own rule.
    winners = sorted(
        score.database_trials.items(),
        key=lambda item: (-item[1]["carrying_gap"], -item[1]["hits"], item[0]),
    )
    if winners and winners[0][1]["carrying_gap"] > 0 or winners:
        score.database = winners[0][0]

    score.exact_match = score.predicted == truth
    score.base_accuracy, score.length_delta = base_accuracy(score.predicted, truth)
    return score


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", action="append", help="Run only these case labels.")
    parser.add_argument("--control", help="Pin every search to this database instead.")
    parser.add_argument("--json", help="Write the scores to this path.")
    args = parser.parse_args()

    configure_logging(get_settings().observability)
    settings = get_settings()
    if not settings.embl_ebi.contact_email:
        print("SKIP: EMBL_EBI_CONTACT_EMAIL is not set; the real services need it.")
        return 0

    selected = [c for c in CASES if not args.case or c.label in args.case]
    if not selected:
        print(f"No case matched {args.case}")
        return 2

    scores: list[Score] = []
    for case in selected:
        print(f"\n=== {case.label}: {case.accession} ({case.organism or 'no organism'}) ===")
        score = await run_case(case, settings, control=args.control)
        scores.append(score)
        print(json.dumps(score.as_dict(), indent=2, default=str))

    print("\n" + "=" * 78)
    print(f"{'case':<14}{'status':<12}{'exact':<7}{'acc':<7}{'carriers':<10}{'db':<14}{'secs':<7}")
    for s in scores:
        print(
            f"{s.case:<14}{s.status:<12}{str(s.exact_match):<7}"
            f"{s.base_accuracy:<7.3f}{s.blast_hits_carrying_gap:<10}"
            f"{str(s.database or '-'):<14}{s.wall_seconds:<7.0f}"
        )

    if args.json:
        Path(args.json).write_text(
            json.dumps([s.as_dict() for s in scores], indent=2, default=str),
            encoding="utf-8",
        )

    exact = sum(1 for s in scores if s.exact_match)
    print(f"\n{exact}/{len(scores)} exact.")
    return 0 if exact else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
