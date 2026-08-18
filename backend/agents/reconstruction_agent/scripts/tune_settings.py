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
import random
import sys
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


def build_case(scenario: Scenario, rng: random.Random) -> tuple[GapContext, Alignment, str]:
    """A gap context, an alignment over it, and the truth it should recover."""
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
    return context, alignment, truth


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


def run_confidence_sweep(seed: int = 20260818) -> list[Outcome]:
    """Score every scenario with the deterministic pipeline."""
    rng = random.Random(seed)
    reasoner = Reasoner(
        ranker=CandidateRanker(),
        validator=ReconstructionValidator(),
        # Threshold is applied afterwards during the sweep, so score with the
        # floor at zero and classify later.
        confidence=ConfidencePolicy(minimum_confidence=0.0),
    )

    outcomes: list[Outcome] = []
    for scenario in scenarios():
        context, alignment, truth = build_case(scenario, rng)
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
            )
        )
    return outcomes


def report_confidence(outcomes: list[Outcome]) -> float:
    """Print precision/recall by threshold and return the recommended value."""
    print("\n" + "=" * 78)
    print("SWEEP 1 - RECONSTRUCTION_MIN_CONFIDENCE")
    print("=" * 78)
    print(f"{len(outcomes)} scenarios; {sum(o.correct for o in outcomes)} recoverable.")
    print(
        "\nA wrong base is worse than no base: a reported reconstruction that is "
        "\nincorrect corrupts an assembly silently, while an unresolved gap is "
        "\nvisibly unresolved. So precision is weighted above recall."
    )
    print(f"\n{'thresh':>7} {'accepted':>9} {'correct':>8} {'precision':>10} "
          f"{'recall':>8} {'F1':>7}")
    print("-" * 78)

    recoverable = sum(1 for o in outcomes if o.correct) or 1
    best_f1, best_threshold = -1.0, 0.55
    at_95: float | None = None
    clean: float | None = None
    clean_recall = 0.0

    for step in range(0, 20):
        threshold = step / 20
        accepted = [o for o in outcomes if o.produced and o.confidence >= threshold]
        correct = sum(1 for o in accepted if o.correct)
        precision = correct / len(accepted) if accepted else 1.0
        recall = correct / recoverable
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

        if f1 > best_f1:
            best_f1, best_threshold = f1, threshold
        if accepted and precision >= 0.95 and at_95 is None:
            at_95 = threshold
        if accepted and precision >= 1.0 and clean is None:
            clean, clean_recall = threshold, recall

        print(f"{threshold:>7.2f} {len(accepted):>9} {correct:>8} {precision:>10.3f} "
              f"{recall:>8.3f} {f1:>7.3f}")

    print("-" * 78)
    print(f"best F1              : {best_threshold:.2f}  (F1={best_f1:.3f})")
    print(f"precision >= 0.95    : {at_95 if at_95 is not None else 'not reached'}")
    print(f"precision == 1.00    : {clean if clean is not None else 'not reached'}"
          f"  (recall {clean_recall:.3f})")

    # The zero-false-positive threshold, not the F1 optimum. F1 treats a wrong
    # base and a missing base as equally bad; in an assembly they are not. A
    # missing base stays visibly missing, a wrong one propagates silently into
    # every downstream analysis.
    recommended = clean if clean is not None else (at_95 or best_threshold)
    print(f"\nRECOMMENDED RECONSTRUCTION_MIN_CONFIDENCE={recommended:.2f}")
    print("Chosen for zero false positives on this grid, not for best F1.")
    print("Caveat: the grid over-represents adversarial 50/50 reference splits,")
    print("so this is a conservative floor rather than a calibrated optimum.")
    return recommended


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
    args = parser.parse_args()

    settings = Settings()
    configure_logging(settings.observability, log_format=LogFormat.JSON)

    outcomes = run_confidence_sweep()
    confidence = report_confidence(outcomes)
    report_failure_modes(outcomes)

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
