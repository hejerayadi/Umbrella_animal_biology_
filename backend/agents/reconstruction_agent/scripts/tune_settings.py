"""Measure the agent's settings against known answers, and recommend values.

Two independent sweeps, because two different things are being tuned:

1. **Confidence threshold** (`RECONSTRUCTION_MIN_CONFIDENCE`) - offline and
   free. The reconstructed bases come from deterministic alignment consensus,
   never from the LLM, so this can be measured exactly against ground truth by
   punching holes in known sequences.

2. **LLM temperature** (`LLM_TEMPERATURE`) - hits Azure. Temperature cannot
   change reconstruction accuracy, because it does not touch the consensus. It
   changes whether the planner emits parseable JSON, whether it picks sensible
   tools, and whether the same input gives the same plan twice.

Run:

    uv run python scripts/tune_settings.py               # both sweeps
    uv run python scripts/tune_settings.py --offline     # skip the Azure calls
    uv run python scripts/tune_settings.py --repeats 5   # more LLM samples
"""
from __future__ import annotations

import argparse
import asyncio
import itertools
import json
import random
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agent import prompts  # noqa: E402
from agent.planning.planner import Planner  # noqa: E402
from agent.reasoning.reasoner import Reasoner  # noqa: E402
from configuration.logging import configure_logging  # noqa: E402
from configuration.runtime import use_selector_event_loop  # noqa: E402
from configuration.settings import LogFormat, Settings  # noqa: E402
from contracts.output import ReconstructionStatus  # noqa: E402
from domain.models import AlignedPair, Alignment, Gap, GapContext  # noqa: E402
from domain.policies import ConfidencePolicy  # noqa: E402
from domain.services import CandidateRanker, ReconstructionValidator  # noqa: E402
from infrastructure.llm.client import AzureOpenAIClient, Message  # noqa: E402

BASES = "ACGT"


# --------------------------------------------------------------------------
# Scenario construction
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Scenario:
    """One synthetic gap with a known correct answer."""

    name: str
    gap_length: int
    reference_count: int
    agreeing_fraction: float
    both_flanks: bool
    flank_divergence: float

    @property
    def label(self) -> str:
        return (
            f"len={self.gap_length:<4} refs={self.reference_count} "
            f"agree={self.agreeing_fraction:.2f} "
            f"flanks={'2' if self.both_flanks else '1'} "
            f"div={self.flank_divergence:.0%}"
        )


def _mutate(sequence: str, rate: float, rng: random.Random) -> str:
    """Substitute `rate` of positions, so references differ in their flanks."""
    if rate <= 0:
        return sequence
    out = []
    for base in sequence:
        if rng.random() < rate:
            out.append(rng.choice([b for b in BASES if b != base]))
        else:
            out.append(base)
    return "".join(out)


def _plurality_fill(pairs: list[AlignedPair], start: int, end: int) -> str:
    """The unweighted per-column plurality over the reference rows.

    Deliberately a local reimplementation rather than a call into
    `CandidateRanker`: this is the yardstick recall is measured against, so it
    must not move when the ranker's voting rule changes. If a future weighted
    voter recovers a case this cannot, that shows up as `correct` exceeding
    `decidable` - which is a result worth seeing, not an error.
    """
    out: list[str] = []
    for column in range(start, end):
        votes = Counter(
            pair.reference_aligned[column]
            for pair in pairs
            if column < len(pair.reference_aligned)
        )
        if not votes:
            continue
        ranked = votes.most_common()
        # A tie is not a plurality. The old yardstick took `most_common`'s
        # insertion order, which in this fixture put the agreeing references
        # first - so it scored ties as winnable purely because of how the
        # scenario was built. That flattered any voter that broke ties the same
        # way, and penalised one that did not.
        if len(ranked) > 1 and ranked[1][1] == ranked[0][1]:
            return ""

        base = ranked[0][0]
        if base != "-":
            out.append(base)
    return "".join(out)


def case_rng(scenario: Scenario, seed: int) -> random.Random:
    """A generator keyed on the scenario's own coordinates.

    One shared `Random` drawn sequentially made every scenario's data depend on
    how many scenarios preceded it, so reordering the grid silently changed the
    results. Seeding per scenario makes each case reproducible on its own.
    """
    return random.Random(
        f"{seed}:{scenario.gap_length}:{scenario.reference_count}:"
        f"{scenario.agreeing_fraction}:{scenario.both_flanks}:{scenario.flank_divergence}"
    )


def build_case(
    scenario: Scenario, rng: random.Random
) -> tuple[GapContext, Alignment, str, bool]:
    """A gap context, an alignment over it, the truth, and whether it is winnable.

    The last element is decided from the fixture alone - never from what the
    pipeline produced - because it is the denominator recall is measured
    against. Deriving it from the outcome (as this script used to) made recall
    self-referential: recovering more cases also raised its own denominator, so
    a real improvement could read as flat.
    """
    flank_length = 120
    left = "".join(rng.choice(BASES) for _ in range(flank_length))
    right = "".join(rng.choice(BASES) for _ in range(flank_length))
    truth = "".join(rng.choice(BASES) for _ in range(scenario.gap_length))

    context = GapContext(
        gap=Gap("gap_1", flank_length, flank_length + scenario.gap_length),
        left_flank=left,
        # A single-flank scenario has nothing usable on the right.
        right_flank=right if scenario.both_flanks else "",
    )

    target_row = left + "-" * scenario.gap_length + (right if scenario.both_flanks else "")
    agreeing = max(1, round(scenario.reference_count * scenario.agreeing_fraction))

    pairs = []
    for index in range(scenario.reference_count):
        # Disagreeing references carry a different filling, not noise: that is
        # what a genuinely divergent homologue looks like.
        filling = truth if index < agreeing else "".join(
            rng.choice([b for b in BASES if b != t]) for t in truth
        )
        row = (
            _mutate(left, scenario.flank_divergence, rng)
            + filling
            + (_mutate(right, scenario.flank_divergence, rng) if scenario.both_flanks else "")
        )
        pairs.append(AlignedPair("target", f"REF_{index}", target_row, row))

    alignment = Alignment(
        gap_id="gap_1",
        pairs=pairs,
        gap_column_start=flank_length,
        gap_column_end=flank_length + scenario.gap_length,
    )
    decidable = (
        _plurality_fill(pairs, flank_length, flank_length + scenario.gap_length) == truth
    )
    return context, alignment, truth, decidable


def scenarios() -> list[Scenario]:
    """The grid. Deliberately includes cases the agent should refuse."""
    grid = itertools.product(
        (8, 30, 100, 400),        # gap length
        (1, 2, 3, 5, 8),          # reference count
        (1.0, 0.75, 0.5),         # fraction agreeing
        (True, False),            # both flanks
        (0.0, 0.05, 0.15),        # flank divergence
    )
    return [
        Scenario(
            name=f"s{index}",
            gap_length=length,
            reference_count=count,
            agreeing_fraction=agree,
            both_flanks=both,
            flank_divergence=divergence,
        )
        for index, (length, count, agree, both, divergence) in enumerate(grid)
    ]


# --------------------------------------------------------------------------
# Sweep 1: confidence threshold (offline)
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Outcome:
    scenario: Scenario
    confidence: float
    correct: bool
    produced: bool
    #: Whether the truth was derivable from the references supplied, decided
    #: from the fixture rather than from what the pipeline managed.
    decidable: bool


def run_confidence_sweep(seed: int = 20260818) -> list[Outcome]:
    """Score every scenario with the deterministic pipeline."""
    reasoner = Reasoner(
        ranker=CandidateRanker(),
        validator=ReconstructionValidator(),
        # Threshold is applied afterwards during the sweep, so score with the
        # floor at zero and classify later.
        confidence=ConfidencePolicy(minimum_confidence=0.0),
    )

    outcomes: list[Outcome] = []
    for scenario in scenarios():
        context, alignment, truth, decidable = build_case(scenario, case_rng(scenario, seed))
        candidates = reasoner.build_candidates(context, alignment, [])
        result = reasoner.finalise(context, candidates)

        produced = (
            result.status is not ReconstructionStatus.UNRESOLVED
            and result.reconstructed_sequence is not None
        )
        outcomes.append(
            Outcome(
                scenario=scenario,
                confidence=result.confidence,
                correct=produced and result.reconstructed_sequence == truth,
                produced=produced,
                decidable=decidable,
            )
        )
    return outcomes


def measure(outcomes: list[Outcome], threshold: float) -> dict[str, float]:
    """Precision, recall and F1 at one threshold.

    Recall is over **every** scenario, not over the ones the pipeline happened
    to get right. The denominator is therefore a fixed property of the grid and
    comparable across versions of the agent - which is the whole point of
    measuring. `recall_decidable` narrows it to the cases an unweighted
    plurality could win, for interpretation only.
    """
    accepted = [o for o in outcomes if o.produced and o.confidence >= threshold]
    correct = sum(1 for o in accepted if o.correct)
    decidable = sum(1 for o in outcomes if o.decidable) or 1

    precision = correct / len(accepted) if accepted else 1.0
    recall = correct / (len(outcomes) or 1)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return {
        "threshold": threshold,
        "accepted": float(len(accepted)),
        "correct": float(correct),
        "precision": precision,
        "recall": recall,
        "recall_decidable": correct / decidable,
        "f1": f1,
    }


def report_confidence(outcomes: list[Outcome], min_precision: float = 0.90) -> float:
    """Print the precision/recall curve and return the recommended threshold."""
    decidable = sum(1 for o in outcomes if o.decidable)

    print("\n" + "=" * 78)
    print("SWEEP 1 - RECONSTRUCTION_MIN_CONFIDENCE")
    print("=" * 78)
    print(f"{len(outcomes)} scenarios; {decidable} winnable by unweighted plurality.")
    print(
        "\nRecall is over all scenarios, so the denominator cannot move when the\n"
        "agent improves. It is therefore comparable across versions, and its\n"
        f"ceiling on this grid is {decidable / len(outcomes):.3f}, not 1.000."
    )
    print(
        f"\nThe operating point is chosen for the highest recall whose precision\n"
        f"still meets {min_precision:.2f}. A wrong base is worse than a missing one -\n"
        "it corrupts an assembly silently - so precision sets the floor, but a\n"
        "tool that refuses two answers in three is not doing its job either."
    )
    print(f"\n{'thresh':>7} {'accepted':>9} {'correct':>8} {'precision':>10} "
          f"{'recall':>8} {'F1':>7}")
    print("-" * 78)

    rows = [measure(outcomes, step / 20) for step in range(0, 20)]
    for row in rows:
        print(f"{row['threshold']:>7.2f} {int(row['accepted']):>9} {int(row['correct']):>8} "
              f"{row['precision']:>10.3f} {row['recall']:>8.3f} {row['f1']:>7.3f}")

    print("-" * 78)

    # Among the thresholds that clear the floor, the best F1 - not the lowest.
    # Taking the lowest maximises recall by definition, but it gives away
    # precision that costs almost no recall to keep: on the current curve the
    # lowest qualifying threshold buys 0.04 recall for 0.04 precision, which is
    # not a trade worth making when a wrong base corrupts an assembly silently.
    qualifying = [r for r in rows if r["accepted"] and r["precision"] >= min_precision]
    qualifying.sort(key=lambda r: (-r["f1"], r["threshold"]))
    best_f1 = max(rows, key=lambda r: r["f1"])
    clean = next((r for r in rows if r["accepted"] and r["precision"] >= 1.0), None)

    if clean:
        print(f"precision == 1.00    : {clean['threshold']:.2f}  "
              f"(recall {clean['recall']:.3f})")
    else:
        print("precision == 1.00    : not reached")
    print(f"best F1              : {best_f1['threshold']:.2f}  (F1={best_f1['f1']:.3f})")

    chosen = qualifying[0] if qualifying else best_f1
    print(f"\nRECOMMENDED RECONSTRUCTION_MIN_CONFIDENCE={chosen['threshold']:.2f}")
    print(f"  precision {chosen['precision']:.3f}  recall {chosen['recall']:.3f}  "
          f"(of winnable: {chosen['recall_decidable']:.3f})")
    if not qualifying:
        print(f"  NO threshold reaches precision {min_precision:.2f}; fell back to best F1.")
    print("Caveat: the grid over-represents adversarial 50/50 reference splits,")
    print("so this is a conservative floor rather than a calibrated optimum.")
    return float(chosen["threshold"])


#: Committed alongside the script, so every change to the agent can be shown
#: against the numbers it started from rather than against a memory of them.
BASELINE_PATH = Path(__file__).resolve().parent / "tuning_baseline.json"


def compare_to_baseline(outcomes: list[Outcome], *, save: bool = False) -> None:
    """Show this run's curve against the committed baseline, and optionally replace it."""
    current = {
        "scenarios": len(outcomes),
        "decidable": sum(1 for o in outcomes if o.decidable),
        "curve": [measure(outcomes, step / 20) for step in range(0, 20)],
    }

    if save:
        BASELINE_PATH.write_text(json.dumps(current, indent=2) + "\n", encoding="utf-8")
        print(f"\nBaseline written to {BASELINE_PATH.name}.")
        return

    if not BASELINE_PATH.exists():
        print(f"\nNo baseline yet. Run with --save-baseline to record {BASELINE_PATH.name}.")
        return

    previous = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    print("\n" + "=" * 78)
    print(f"AGAINST BASELINE ({BASELINE_PATH.name})")
    print("=" * 78)
    print(f"{'thresh':>7} {'precision':>21} {'recall':>21}")
    print("-" * 78)

    by_threshold = {row["threshold"]: row for row in previous.get("curve", [])}
    for row in current["curve"]:
        before = by_threshold.get(row["threshold"])
        if before is None:
            continue
        d_precision = row["precision"] - before["precision"]
        d_recall = row["recall"] - before["recall"]
        if abs(d_precision) < 1e-9 and abs(d_recall) < 1e-9:
            continue
        print(
            f"{row['threshold']:>7.2f} "
            f"{before['precision']:>8.3f} -> {row['precision']:>6.3f} ({d_precision:+.3f}) "
            f"{before['recall']:>8.3f} -> {row['recall']:>6.3f} ({d_recall:+.3f})"
        )
    print("-" * 78)


def report_discrimination(outcomes: list[Outcome]) -> None:
    """Whether the confidence score separates correct answers from wrong ones.

    This is the measurement that matters most while the scoring is being
    reworked. A threshold can only trade recall for precision if the score
    ranks correct reconstructions above incorrect ones; if the two populations
    overlap, raising the threshold discards good answers and bad ones in equal
    proportion and buys nothing.

    Reported as the separation between the two means, and as the probability
    that a randomly chosen correct answer outscores a randomly chosen wrong one
    (the rank statistic behind AUC - 0.5 is no signal, 1.0 is perfect ranking).
    """
    produced = [o for o in outcomes if o.produced]
    right = [o.confidence for o in produced if o.correct]
    wrong = [o.confidence for o in produced if not o.correct]

    print()
    print("=" * 78)
    print("DOES CONFIDENCE PREDICT CORRECTNESS?")
    print("=" * 78)

    if not right or not wrong:
        print("Only one population present; separation cannot be measured.")
        return

    mean_right = sum(right) / len(right)
    mean_wrong = sum(wrong) / len(wrong)

    wins = sum(
        1.0 if r > w else 0.5 if r == w else 0.0
        for r in right
        for w in wrong
    )
    auc = wins / (len(right) * len(wrong))

    print(f"correct   : n={len(right):>3}  mean confidence {mean_right:.3f}")
    print(f"incorrect : n={len(wrong):>3}  mean confidence {mean_wrong:.3f}")
    print(f"separation: {mean_right - mean_wrong:+.3f}")
    print(f"ranking   : {auc:.3f}  (0.5 = the score carries no signal)")

    if auc < 0.65:
        print()
        print(
            "The score barely ranks correct above incorrect, so the threshold "
            "is a blunt instrument: it discards good answers at nearly the same "
            "rate as bad ones. Improving the score beats moving the threshold."
        )


def report_failure_modes(outcomes: list[Outcome]) -> None:
    """Where the pipeline produces a confident wrong answer - the dangerous case."""
    print("\n" + "=" * 78)
    print("CONFIDENT-BUT-WRONG CASES (confidence >= 0.55 and incorrect)")
    print("=" * 78)

    bad = [o for o in outcomes if o.produced and not o.correct and o.confidence >= 0.55]
    if not bad:
        print("None. No scenario produced a wrong answer above the current threshold.")
        return

    for outcome in sorted(bad, key=lambda o: -o.confidence)[:15]:
        print(f"  conf={outcome.confidence:.3f}  {outcome.scenario.label}")
    print(f"\n{len(bad)} of {len(outcomes)} scenarios. These are what the threshold must exclude.")


# --------------------------------------------------------------------------
# Sweep 2: LLM temperature (live)
# --------------------------------------------------------------------------


PLANNING_STATES = [
    {
        "name": "fresh gaps, no evidence",
        "instruction": "Reconstruct the unresolved regions of this mammoth genome fragment.",
        "organism": "Mammuthus primigenius",
        "gaps": [
            {"gap_id": "gap_1", "length": 120, "left_flank_length": 500,
             "right_flank_length": 500, "has_both_flanks": True,
             "reference_count": 0, "status": "open"},
        ],
    },
    {
        "name": "references gathered, needs alignment",
        "instruction": "Fill the gaps using the references already retrieved.",
        "organism": "Loxodonta africana",
        "gaps": [
            {"gap_id": "gap_1", "length": 60, "left_flank_length": 400,
             "right_flank_length": 400, "has_both_flanks": True,
             "reference_count": 6, "status": "open"},
        ],
    },
    {
        "name": "unreconstructable gap",
        "instruction": "Reconstruct this region.",
        "organism": "Testus organismus",
        "gaps": [
            {"gap_id": "gap_1", "length": 4000, "left_flank_length": 12,
             "right_flank_length": 0, "has_both_flanks": False,
             "reference_count": 0, "status": "open"},
        ],
    },
]

TOOL_CATALOGUE = [
    {"name": "ncbi_search", "description": "Search NCBI for reference sequences by organism.",
     "estimated_seconds": 3.0},
    {"name": "blast_search", "description": "BLAST homology search over a gap's flanks. Slow.",
     "estimated_seconds": 60.0},
    {"name": "mafft_align", "description": "Align references to locate the gap's columns. Slow.",
     "estimated_seconds": 45.0},
    {"name": "evolutionary_context", "description": "Rank organisms by relatedness. Fast.",
     "estimated_seconds": 0.1},
]
KNOWN_TOOLS = {tool["name"] for tool in TOOL_CATALOGUE}


@dataclass
class TemperatureResult:
    temperature: float
    calls: int = 0
    parsed: int = 0
    valid_tools: int = 0
    #: state name -> the plan produced on each repeat of that state.
    plans_by_state: dict[str, list[str]] | None = None

    def __post_init__(self) -> None:
        if self.plans_by_state is None:
            self.plans_by_state = {}

    @property
    def parse_rate(self) -> float:
        return self.parsed / self.calls if self.calls else 0.0

    @property
    def tool_validity(self) -> float:
        return self.valid_tools / self.parsed if self.parsed else 0.0

    @property
    def determinism(self) -> float:
        """How often re-asking the same question gives the same plan.

        Measured within each state and then averaged. Pooling every state's
        plans together would be meaningless: three states answered perfectly
        consistently still yield three different plans, which reads as 1/3.
        """
        assert self.plans_by_state is not None
        per_state = [
            max(plans.count(plan) for plan in set(plans)) / len(plans)
            for plans in self.plans_by_state.values()
            if plans
        ]
        return sum(per_state) / len(per_state) if per_state else 0.0


async def run_temperature_sweep(
    settings: Settings, temperatures: list[float], repeats: int
) -> list[TemperatureResult]:
    """Ask the real model to plan, at each temperature, and score the output."""
    results: list[TemperatureResult] = []

    for temperature in temperatures:
        client = AzureOpenAIClient(
            base_url=settings.azure.openai_base_url or "",
            deployment=settings.azure.openai_deployment or "",
            api_key=settings.azure.openai_api_key or "",
            api_version=settings.azure.openai_api_version,
            temperature=temperature,
            max_tokens=settings.llm.max_tokens,
        )
        planner = Planner(client, sorted(KNOWN_TOOLS))
        result = TemperatureResult(temperature=temperature)

        try:
            for state in PLANNING_STATES:
                per_state: list[str] = []
                for _ in range(repeats):
                    messages = [
                        Message("system", prompts.planner_system()),
                        Message(
                            "user",
                            prompts.planner_user(
                                instruction=state["instruction"],
                                organism=state["organism"],
                                gaps=state["gaps"],
                                tools=TOOL_CATALOGUE,
                                prior_critiques=[],
                            ),
                        ),
                    ]
                    result.calls += 1
                    try:
                        reply = await client.complete(messages)
                    except Exception as error:  # noqa: BLE001
                        print(f"    call failed at t={temperature}: {str(error)[:90]}")
                        continue

                    steps = planner._parse(reply)
                    if steps:
                        result.parsed += 1
                        if all(step.tool in KNOWN_TOOLS for step in steps):
                            result.valid_tools += 1
                        per_state.append(
                            " | ".join(f"{s.tool}:{s.gap_id}" for s in steps)
                        )
                assert result.plans_by_state is not None
                result.plans_by_state[str(state["name"])] = per_state
        finally:
            await client.aclose()

        results.append(result)
        print(
            f"  t={temperature:<4} parsed {result.parsed}/{result.calls}  "
            f"tools_ok {result.tool_validity:.2f}  determinism {result.determinism:.2f}"
        )

    return results


def report_temperature(results: list[TemperatureResult]) -> float:
    print("\n" + "=" * 78)
    print("SWEEP 2 - LLM_TEMPERATURE")
    print("=" * 78)
    print(
        "Temperature cannot change reconstruction accuracy - the bases come from"
        "\nalignment consensus, not the model. What it changes is whether the plan"
        "\nparses, names real tools, and repeats."
    )
    print(f"\n{'temp':>6} {'calls':>6} {'parsed':>7} {'parse%':>8} "
          f"{'tools_ok':>9} {'determinism':>12}")
    print("-" * 78)

    for result in results:
        print(f"{result.temperature:>6.1f} {result.calls:>6} {result.parsed:>7} "
              f"{result.parse_rate:>8.2f} {result.tool_validity:>9.2f} "
              f"{result.determinism:>12.2f}")

    print("-" * 78)
    usable = [r for r in results if r.parse_rate >= 0.99 and r.tool_validity >= 0.99]
    if usable:
        best = max(usable, key=lambda r: (r.determinism, -r.temperature))
        print(f"\nRECOMMENDED LLM_TEMPERATURE={best.temperature:.1f}")
        print("Highest determinism among temperatures that always parsed and always")
        print("named real tools. Reconstruction planning should be reproducible.")
        return best.temperature

    print("\nNo temperature parsed reliably. Keep LLM_TEMPERATURE=0.0 and rely on")
    print("the deterministic planner fallback.")
    return 0.0


# --------------------------------------------------------------------------


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline", action="store_true", help="Skip the Azure sweep.")
    parser.add_argument("--repeats", type=int, default=3, help="LLM samples per state.")
    parser.add_argument(
        "--min-precision",
        type=float,
        default=0.90,
        help="Precision floor the recommended threshold must meet (default 0.90).",
    )
    parser.add_argument(
        "--save-baseline",
        action="store_true",
        help=f"Overwrite {BASELINE_PATH.name} with this run's curve.",
    )
    args = parser.parse_args()

    settings = Settings()
    configure_logging(settings.observability, log_format=LogFormat.JSON)

    outcomes = run_confidence_sweep()
    confidence = report_confidence(outcomes, min_precision=args.min_precision)
    report_discrimination(outcomes)
    report_failure_modes(outcomes)
    compare_to_baseline(outcomes, save=args.save_baseline)

    temperature = 0.0
    if not args.offline:
        if not settings.azure.configured:
            print("\nSkipping the temperature sweep: Azure is not configured.")
        else:
            print("\nRunning the temperature sweep against Azure "
                  f"({settings.azure.openai_deployment})...")
            results = await run_temperature_sweep(
                settings, [0.0, 0.3, 0.7, 1.0], args.repeats
            )
            temperature = report_temperature(results)

    print("\n" + "=" * 78)
    print("RECOMMENDED .env SETTINGS")
    print("=" * 78)
    print(f"RECONSTRUCTION_MIN_CONFIDENCE={confidence:.2f}")
    if not args.offline:
        print(f"LLM_TEMPERATURE={temperature:.1f}")
    return 0


if __name__ == "__main__":
    use_selector_event_loop()
    raise SystemExit(asyncio.run(main()))
