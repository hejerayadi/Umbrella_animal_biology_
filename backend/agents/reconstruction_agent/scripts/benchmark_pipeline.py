"""End-to-end benchmark: every stage the real agent runs, none of them bypassed.

`tune_settings.py` measures the scoring layer alone. It hands `Reasoner` an
alignment built by hand, so it cannot see retrieval, gap-column location, or
filtering - which is exactly where the agent was losing recall. A change that
fixed the evidence path was invisible there.

This runs the whole chain instead:

    gap -> planning -> BLAST -> sequence retrieval -> filtering
        -> MAFFT -> reasoning -> validation -> critic -> result

The external services are simulated, the agent is not. The real
`BlastSearchTool`, `NCBISearchTool`, `MafftAlignmentTool` and
`EvolutionaryContextTool` run against fake transports, so their parsing,
retrieval and mapping logic is under test. Only the network and the BLAST/MAFFT
algorithms themselves are stood in for - and those are simulated from a known
world, so ground truth is exact.

Run:

    uv run python scripts/benchmark_pipeline.py
    uv run python scripts/benchmark_pipeline.py --threshold 0.5
"""
from __future__ import annotations

import argparse
import asyncio
import itertools
import json
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from application.result_builder import ResultBuilder  # noqa: E402
from application.run_agent import AgentRunner  # noqa: E402
from configuration.logging import configure_logging  # noqa: E402
from configuration.settings import (  # noqa: E402
    BudgetSettings,
    ContinuationSettings,
    EMBLEBISettings,
    LLMSettings,
    LogFormat,  # noqa: E402
    NCBISettings,
    ObservabilitySettings,
    Settings,
)
from contracts.output import ReconstructionStatus  # noqa: E402
from domain.models import Sequence  # noqa: E402
from domain.models.sequence import reverse_complement  # noqa: E402
from observability.events import CollectingEmitter  # noqa: E402
from tools.blast.advisor import DatabaseAdvisor  # noqa: E402
from tools.blast.catalogue import EbiDatabaseCatalogue  # noqa: E402
from tools.blast.tool import BlastSearchTool  # noqa: E402
from tools.evo.tool import Evo2PlausibilityTool, EvolutionaryContextTool  # noqa: E402
from tools.mafft.tool import MafftAlignmentTool  # noqa: E402
from tools.ncbi.tool import NCBISearchTool  # noqa: E402
from tools.registry import ToolRegistry  # noqa: E402

BASES = "ACGT"
FLANK = 300


# --------------------------------------------------------------------------
# A world the simulated services answer from
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Record:
    """One database record, as the simulated NCBI and BLAST both see it."""

    accession: str
    organism: str
    description: str
    residues: str
    #: Where the homologous region sits inside `residues`, 1-based inclusive.
    start: int
    identity: float
    strand: int
    #: Whether this record carries the true fill. A pseudogene does not.
    carries_truth: bool


@dataclass
class World:
    """One reconstruction problem plus every record a search could return."""

    target: Sequence
    truth: str
    gap_start: int
    gap_length: int
    records: list[Record] = field(default_factory=list)

    def by_accession(self, accession: str) -> Record | None:
        return next((r for r in self.records if r.accession == accession), None)


def _mutate(sequence: str, rate: float, rng: random.Random) -> str:
    if rate <= 0:
        return sequence
    return "".join(
        rng.choice([b for b in BASES if b != base]) if rng.random() < rate else base
        for base in sequence
    )


@dataclass(frozen=True, slots=True)
class Case:
    """One point on the benchmark grid."""

    gap_length: int
    reference_count: int
    agreeing_fraction: float
    divergence: float
    pseudogene_fraction: float
    minus_strand: bool

    @property
    def label(self) -> str:
        return (
            f"len={self.gap_length:<4} refs={self.reference_count} "
            f"agree={self.agreeing_fraction:.2f} div={self.divergence:.0%} "
            f"pseudo={self.pseudogene_fraction:.0%} "
            f"strand={'-' if self.minus_strand else '+'}"
        )


def build_world(case: Case, rng: random.Random) -> World:
    """A target with one gap, and the records a search over it would find."""
    left = "".join(rng.choice(BASES) for _ in range(FLANK))
    right = "".join(rng.choice(BASES) for _ in range(FLANK))
    truth = "".join(rng.choice(BASES) for _ in range(case.gap_length))

    target = Sequence.parse(
        "bench_target",
        left + "N" * case.gap_length + right,
        organism="Testus organismus",
    )

    agreeing = max(1, round(case.reference_count * case.agreeing_fraction))
    pseudogenes = round(case.reference_count * case.pseudogene_fraction)
    records: list[Record] = []

    for index in range(case.reference_count):
        is_pseudogene = index >= case.reference_count - pseudogenes
        carries_truth = index < agreeing and not is_pseudogene

        fill = (
            truth
            if carries_truth
            else "".join(rng.choice([b for b in BASES if b != t]) for t in truth)
        )
        # Records are longer than the homologous region, so the retrieval step
        # has to locate it rather than being handed the whole thing.
        lead = "".join(rng.choice(BASES) for _ in range(120))
        tail = "".join(rng.choice(BASES) for _ in range(120))
        homologous = (
            _mutate(left, case.divergence, rng) + fill + _mutate(right, case.divergence, rng)
        )
        residues = lead + homologous + tail
        strand = -1 if case.minus_strand and index % 2 == 0 else 1

        records.append(
            Record(
                accession=f"REF_{index}",
                organism="Testus organismus" if index % 3 == 0 else "Testus cousinus",
                description=(
                    "pseudogene, partial sequence" if is_pseudogene else "complete cds"
                ),
                # Stored on the strand the database holds it on.
                residues=reverse_complement(residues) if strand < 0 else residues,
                start=len(lead) + 1,
                identity=max(0.0, 1.0 - case.divergence),
                strand=strand,
                carries_truth=carries_truth,
            )
        )

    return World(
        target=target,
        truth=truth,
        gap_start=FLANK,
        gap_length=case.gap_length,
        records=records,
    )


# --------------------------------------------------------------------------
# Simulated services - fake transport, real tools on top
# --------------------------------------------------------------------------


class FakeBlastClient:
    """Answers with HSPs shaped the way a real gap search produces them.

    Two HSPs per hit, one per flank, bracketing the segment the query does not
    contain. That shape is the whole reason the retrieval step exists.
    """

    def __init__(self, world: World) -> None:
        self._world = world
        self.submissions = 0

    async def submit(self, sequence: str, **_: object) -> str:
        self.submissions += 1
        return "blast-job"

    async def result(self, job_id: str, **_: object) -> str:
        hits = []
        for record in self._world.records:
            left_from = record.start
            left_to = record.start + FLANK - 1
            right_from = left_to + self._world.gap_length + 1
            right_to = right_from + FLANK - 1
            strand = "plus/minus" if record.strand < 0 else "plus/plus"

            hits.append(
                {
                    "hit_acc": record.accession,
                    "hit_os": record.organism,
                    "hit_desc": record.description,
                    "hit_len": len(record.residues),
                    "hit_hsps": [
                        {
                            "hsp_identity": record.identity * 100,
                            "hsp_expect": 1e-40,
                            "hsp_bit_score": 500.0,
                            "hsp_align_len": FLANK,
                            "hsp_hit_from": left_from,
                            "hsp_hit_to": left_to,
                            "hsp_strand": strand,
                        },
                        {
                            "hsp_identity": record.identity * 100,
                            "hsp_expect": 1e-38,
                            "hsp_bit_score": 480.0,
                            "hsp_align_len": FLANK,
                            "hsp_hit_from": right_from,
                            "hsp_hit_to": right_to,
                            "hsp_strand": strand,
                        },
                    ],
                }
            )
        return json.dumps({"hits": hits})


class FakeNCBIClient:
    """Serves regions and searches out of the same world BLAST reports on."""

    def __init__(self, world: World) -> None:
        self._world = world
        self.region_requests = 0

    async def search(self, term: str, **kwargs: object) -> list[str]:
        return [record.accession for record in self._world.records]

    async def fetch_fasta(self, identifiers: list[str], **_: object) -> str:
        chunks = []
        for accession in identifiers:
            record = self._world.by_accession(accession)
            if record is not None:
                chunks.append(f">{record.accession} {record.organism} {record.description}")
                chunks.append(record.residues)
        return "\n".join(chunks) + "\n"

    async def fetch_region(
        self, identifier: str, start: int, stop: int, *, database: str = "nuccore", strand: int = 1
    ) -> str:
        self.region_requests += 1
        record = self._world.by_accession(identifier)
        if record is None:
            return ""

        residues = record.residues[max(0, start - 1) : stop]
        if strand < 0:
            residues = reverse_complement(residues)
        return f">{identifier} region\n{residues}\n"

    async def aclose(self) -> None:
        return None


class FakeMafftClient:
    """Aligns by construction, since the world knows where everything sits.

    The real `to_alignment` mapper parses this, and the real gap-column locator
    walks it - so the stage under test is the agent's, not an aligner's.
    """

    def __init__(self, world: World) -> None:
        self._world = world
        self._fasta = ""

    async def submit(self, fasta: str, **_: object) -> str:
        self._fasta = fasta
        return "mafft-job"

    async def result(self, job_id: str, **_: object) -> str:
        records = _parse_fasta(self._fasta)
        if not records:
            return ""

        target_id, target = records[0]
        width = self._world.gap_length
        # The target is the two flanks joined; the aligner inserts the columns
        # the references have and it does not.
        aligned_target = target[:FLANK] + "-" * width + target[FLANK:]

        rows = [f">{target_id}", aligned_target]
        for name, residues in records[1:]:
            rows.append(f">{name}")
            rows.append(_align_reference(residues, width))
        return "\n".join(rows) + "\n"


def _parse_fasta(raw: str) -> list[tuple[str, str]]:
    records: list[tuple[str, str]] = []
    name, chunks = None, []
    for line in raw.splitlines():
        if line.startswith(">"):
            if name is not None:
                records.append((name, "".join(chunks)))
            name, chunks = line[1:].strip().split()[0], []
        elif line.strip():
            chunks.append(line.strip())
    if name is not None:
        records.append((name, "".join(chunks)))
    return records


def _align_reference(residues: str, width: int) -> str:
    """Place a retrieved region against the target's two flanks.

    The retrieved region is `flank + fill + flank` with some slack on either
    side from the fetch margin. The flanks anchor it; whatever sits between
    them is what the reference contributes across the gap.
    """
    if len(residues) < 2 * FLANK + width:
        # Too short to carry a fill - aligns to the flanks with nothing between.
        padded = residues.ljust(2 * FLANK, "-")
        return padded[:FLANK] + "-" * width + padded[FLANK : 2 * FLANK]

    offset = (len(residues) - (2 * FLANK + width)) // 2
    body = residues[offset : offset + 2 * FLANK + width]
    return body


class FakeEvo2Client:
    """A genome model that is informative but not an oracle.

    Predicts the true continuation with a per-base accuracy well short of 1.0,
    which is what a real model does on real sequence. Making it perfect would
    turn every contested gap into a lookup and measure nothing; making it
    useless would leave arbitration untestable. `accuracy` is the dial.
    """

    def __init__(self, world: World, accuracy: float = 0.75) -> None:
        self._world = world
        self._accuracy = accuracy
        self.calls = 0

    async def generate(self, prompt: str, *, num_tokens: int, **_: object):
        from infrastructure.nvidia.client import Continuation

        self.calls += 1
        rng = random.Random(f"evo:{prompt[-40:]}:{num_tokens}")
        predicted = "".join(
            base if rng.random() < self._accuracy
            else rng.choice([b for b in BASES if b != base])
            for base in self._world.truth[:num_tokens]
        )
        return Continuation(
            sequence=predicted,
            sampled_probs=[self._accuracy] * len(predicted),
        )

    async def aclose(self) -> None:
        return None


# --------------------------------------------------------------------------
# Running one case through the real agent
# --------------------------------------------------------------------------


def settings_for(threshold: float) -> Settings:
    return Settings(
        _env_file=None,  # type: ignore[call-arg]
        RECONSTRUCTION_MIN_CONFIDENCE=threshold,
        # The per-tool log is worth reading for one case and unreadable for
        # a hundred; the metrics are the output here.
        observability=ObservabilitySettings(_env_file=None, LOG_LEVEL="WARNING"),  # type: ignore[call-arg]
        llm=LLMSettings(_env_file=None, LLM_PROVIDER="none"),  # type: ignore[call-arg]
        ncbi=NCBISettings(_env_file=None),  # type: ignore[call-arg]
        embl_ebi=EMBLEBISettings(_env_file=None),  # type: ignore[call-arg]
        budgets=BudgetSettings(_env_file=None),  # type: ignore[call-arg]
        continuation=ContinuationSettings(_env_file=None),  # type: ignore[call-arg]
    )


@dataclass
class Trace:
    """What each stage of one run produced, for per-stage measurement."""

    case: Case
    retrieved: bool = False
    aligned: bool = False
    candidate: bool = False
    accepted: bool = False
    produced: bool = False
    correct: bool = False
    confidence: float = 0.0
    #: True when at least one record in the world carried the true fill, so the
    #: answer was there to be found. The denominator, from the world not the run.
    winnable: bool = False


class _FakeTaxonomy:
    """Places the synthetic organisms in a real division, with no network.

    The world's species are invented, so NCBI cannot place them. Returning a
    mammalian lineage lets the harness exercise the real discovery path -
    prior, catalogue validation, database choice - rather than routing around
    it, which is the part most worth regression-testing.
    """

    async def lineage(self, organism: str) -> tuple[str, ...]:
        return ("Eukaryota", "Metazoa", "Chordata", "Mammalia", organism)


def _advisor() -> DatabaseAdvisor:
    """Discovery wired to the offline snapshot: no EBI call, real validation."""
    return DatabaseAdvisor(catalogue=EbiDatabaseCatalogue(None), taxonomy=_FakeTaxonomy())


async def run_case(case: Case, threshold: float, seed: int) -> Trace:
    """One case, end to end, through the real graph and the real tools."""
    rng = random.Random(f"{seed}:{case.label}")
    world = build_world(case, rng)
    trace = Trace(case=case, winnable=any(r.carries_truth for r in world.records))

    ncbi = FakeNCBIClient(world)
    registry = ToolRegistry(
        [
            NCBISearchTool(ncbi),  # type: ignore[arg-type]
            BlastSearchTool(FakeBlastClient(world), ncbi),  # type: ignore[arg-type]
            MafftAlignmentTool(FakeMafftClient(world)),  # type: ignore[arg-type]
            EvolutionaryContextTool(),
            # The arbitration path: without it a contested gap has two
            # hypotheses and nothing to choose between them.
            Evo2PlausibilityTool(FakeEvo2Client(world)),  # type: ignore[arg-type]
        ]
    )

    settings = settings_for(threshold)
    runner = AgentRunner(
        settings, registry, CollectingEmitter(), databases=_advisor()
    )

    outcome = await runner.run_slice(
        run_id="bench",
        trace_id=f"bench-{case.label}",
        instruction="Reconstruct the unresolved region.",
        target=world.target,
        organism="Testus organismus",
        requested_organisms=[],
    )

    state: dict[str, Any] = dict(outcome.state)
    references = (state.get("references") or {}).get("gap_1") or []
    trace.retrieved = any(reference.has_sequence for reference in references)

    alignment = (state.get("alignments") or {}).get("gap_1")
    trace.aligned = bool(alignment is not None and getattr(alignment, "spans_gap", False))
    trace.candidate = bool((state.get("candidates") or {}).get("gap_1"))

    # The same builder the service uses, so the benchmark scores exactly
    # what the orchestrator would have received.
    result = ResultBuilder().build(outcome.state)
    gap = next((g for g in result.gaps if g.gap_id == "gap_1"), None)
    if gap is not None:
        trace.confidence = gap.confidence
        trace.produced = gap.reconstructed_sequence is not None
        trace.accepted = gap.status is ReconstructionStatus.RECONSTRUCTED
        trace.correct = gap.reconstructed_sequence == world.truth

    return trace


def cases() -> list[Case]:
    """A grid spanning the ways real evidence goes wrong."""
    grid = itertools.product(
        (30, 120),            # gap length
        (2, 4, 8),            # references found
        (1.0, 0.75, 0.5),     # fraction carrying the true fill
        (0.0, 0.05),          # flank divergence
        (0.0, 0.25),          # fraction that are pseudogenes
        (False, True),        # some hits on the minus strand
    )
    return [Case(*values) for values in grid]


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------


def report(traces: list[Trace], threshold: float) -> dict[str, float]:
    """Per-stage rates, so a loss can be attributed to the stage that caused it."""
    total = len(traces) or 1
    winnable = [t for t in traces if t.winnable] or traces

    accepted = [t for t in traces if t.accepted]
    correct_accepted = sum(1 for t in accepted if t.correct)

    metrics = {
        "retrieval_recall": sum(t.retrieved for t in traces) / total,
        "alignment_success_rate": sum(t.aligned for t in traces) / total,
        "candidate_recall": sum(t.candidate for t in traces) / total,
        "acceptance_recall": len(accepted) / total,
        "coverage": sum(t.produced for t in traces) / total,
        "precision": correct_accepted / len(accepted) if accepted else 1.0,
        "recall": correct_accepted / len(winnable),
        "f1": 0.0,
    }
    precision, recall = metrics["precision"], metrics["recall"]
    metrics["f1"] = (
        2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    )

    print("=" * 78)
    print(f"END-TO-END PIPELINE - threshold {threshold:.2f}")
    print("=" * 78)
    print(f"{len(traces)} cases; {sum(t.winnable for t in traces)} winnable from the evidence.")
    print()
    print("Each rate is over all cases, so a drop names the stage that lost it.")
    print()
    for name in (
        "retrieval_recall",
        "alignment_success_rate",
        "candidate_recall",
        "acceptance_recall",
        "coverage",
    ):
        print(f"  {name:<24} {metrics[name]:.3f}")
    print()
    print(f"  {'precision':<24} {metrics['precision']:.3f}")
    print(f"  {'recall':<24} {metrics['recall']:.3f}")
    print(f"  {'f1':<24} {metrics['f1']:.3f}")

    lost = [t for t in traces if t.winnable and not t.correct]
    if lost:
        print(f"\n{len(lost)} winnable case(s) not recovered. First few:")
        for trace in lost[:8]:
            stage = (
                "retrieval" if not trace.retrieved
                else "alignment" if not trace.aligned
                else "consensus" if not trace.candidate
                else "threshold" if not trace.accepted
                else "wrong fill"
            )
            print(f"  [{stage:<10}] conf={trace.confidence:.3f}  {trace.case.label}")

    return metrics


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threshold", type=float, default=0.65)
    parser.add_argument("--seed", type=int, default=20260819)
    args = parser.parse_args()

    settings = settings_for(args.threshold)
    # The per-tool log is worth reading for one case and unreadable for a
    # hundred; the metrics are the output here.
    configure_logging(settings.observability, log_format=LogFormat.JSON)

    grid = cases()
    traces = [await run_case(case, args.threshold, args.seed) for case in grid]
    report(traces, args.threshold)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
