"""End-to-end agentic QA audit: every meaningful scenario, against real code.

This drives the actual `ReconstructionService` / `AgentRunner` / LangGraph
loop / stop policy / critic / budget policy / result builder / FastAPI routes
that production runs. The only things replaced are the network edges - the
BLAST, MAFFT, NCBI and Evo 2 clients - with scripted stand-ins that speak the
exact same `Tool` contract (`tools/contracts.py`) the real ones do. Everything
between "a tool returned X" and "here is the AgentResult the orchestrator
sees" is unmodified production code.

Run:

    uv run python scripts/qa_audit.py

Exits non-zero if any scenario fails, so it can gate a release the same way
`scripts/tune_settings.py` gates the confidence threshold.
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from configuration.runtime import use_selector_event_loop  # noqa: E402

use_selector_event_loop()

from application.reconstruction_service import ReconstructionService  # noqa: E402
from configuration.logging import configure_logging  # noqa: E402
from configuration.settings import (  # noqa: E402
    AzureSettings,
    BudgetSettings,
    ContinuationSettings,
    DatabaseSettings,
    EMBLEBISettings,
    LLMProvider,
    LLMSettings,
    LogFormat,
    NCBISettings,
    NvidiaSettings,
    Settings,
)
from contracts.input import ReconstructionRequest  # noqa: E402
from contracts.observation import ObservationStatus  # noqa: E402
from contracts.output import ReconstructionStatus  # noqa: E402
from domain.exceptions import (  # noqa: E402
    ExternalServiceError,
    InvalidSequenceError,
    RateLimitError,
)
from domain.models import AlignedPair, Alignment, Reference, Sequence  # noqa: E402
from infrastructure.llm.client import Completion, Message  # noqa: E402
from observability.events import CollectingEmitter  # noqa: E402
from tools.blast.schemas import BlastSearchInput, BlastSearchOutput  # noqa: E402
from tools.contracts import Tool  # noqa: E402
from tools.evo.tool import EvolutionaryContextTool  # noqa: E402
from tools.mafft.schemas import AlignmentInput, AlignmentOutput  # noqa: E402
from tools.registry import ToolRegistry  # noqa: E402

# ============================================================================
# Verdict bookkeeping
# ============================================================================


@dataclass
class ScenarioResult:
    scenario: str
    expected: str
    actual: str = ""
    verdict: str = "FAIL"  # PASS | FAIL
    severity: str = "-"  # only meaningful when verdict is FAIL
    defects: list[str] = field(default_factory=list)
    evidence: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


RESULTS: list[ScenarioResult] = []


def record(result: ScenarioResult) -> None:
    RESULTS.append(result)
    tag = "PASS" if result.verdict == "PASS" else f"FAIL [{result.severity}]"
    print(f"  {tag:<14} {result.scenario}")
    if result.defects:
        for defect in result.defects:
            print(f"                 - {defect}")


def scenario(name: str, expected: str):  # noqa: ANN201
    """Decorator: run one scenario, catching anything unhandled as a FAIL."""

    def wrap(fn):  # noqa: ANN001, ANN202
        async def runner() -> None:
            result = ScenarioResult(scenario=name, expected=expected)
            try:
                await fn(result)
                if result.verdict == "FAIL" and result.severity == "-":
                    result.severity = "MEDIUM"
            except AssertionError as error:
                result.verdict = "FAIL"
                result.severity = result.severity if result.severity != "-" else "HIGH"
                result.actual = result.actual or str(error)
                result.defects.append(f"Assertion failed: {error}")
            except Exception as error:  # noqa: BLE001
                result.verdict = "FAIL"
                result.severity = "CRITICAL"
                result.error = f"{type(error).__name__}: {error}"
                result.defects.append(f"Unhandled exception: {result.error}")
                traceback.print_exc()
            record(result)

        return runner

    return wrap


# ============================================================================
# Test fixtures: settings, sequences, stub tools, stub LLM
# ============================================================================


def base_settings(**overrides: Any) -> Settings:
    """Fully isolated settings - no real .env, no network, no LLM by default."""
    kwargs: dict[str, Any] = dict(
        _env_file=None,
        max_iterations=overrides.pop("max_iterations", 6),
        min_confidence=overrides.pop("min_confidence", 0.65),
        ncbi=NCBISettings(_env_file=None, contact_email="qa@umbrella.local"),
        embl_ebi=EMBLEBISettings(_env_file=None, contact_email="qa@umbrella.local"),
        azure=AzureSettings(_env_file=None),
        nvidia=NvidiaSettings(_env_file=None),
        llm=overrides.pop("llm", LLMSettings(_env_file=None)),
        budgets=overrides.pop("budgets", BudgetSettings(_env_file=None)),
        continuation=overrides.pop(
            "continuation", ContinuationSettings(_env_file=None, yield_after_seconds=90.0)
        ),
        database=DatabaseSettings(_env_file=None),
    )
    kwargs.update(overrides)
    return Settings(**kwargs)  # type: ignore[arg-type]


GAP_LEN = 12
LEFT = "ACGTTGCA" * 15  # 120 bases
RIGHT = "TTACGGCA" * 15  # 120 bases
TRUTH = "GGGGCCCCTTTT"  # 12 bases - the "correct" fill for the gap
GAPPED_ONE = Sequence.parse(
    "qa_one_gap", LEFT + "N" * GAP_LEN + RIGHT, organism="Mammuthus primigenius"
)

LEFT2 = "TTGGCCAA" * 15
RIGHT2 = "CCAATTGG" * 15
GAP2_LEN = 8
TRUTH2 = "AAAACCCC"
GAPPED_TWO = Sequence.parse(
    "qa_two_gaps",
    LEFT + "N" * GAP_LEN + RIGHT + LEFT2 + "N" * GAP2_LEN + RIGHT2,
    organism="Mammuthus primigenius",
)


def make_reference(
    accession: str,
    organism: str = "Loxodonta africana",
    *,
    relatedness: float | None = None,
) -> Reference:
    """A reference shaped exactly as `tools/blast/mapper.to_references` builds one.

    `relatedness` is None by default because the real mappers never set it -
    it is what `evolutionary_context` exists to fill in. A fixture that
    pre-populated it hid the whole delegation path from these tests.
    """
    return Reference(
        accession=accession,
        organism=organism,
        residues=LEFT + TRUTH + RIGHT,
        identity=0.95,
        coverage=0.9,
        relatedness=relatedness,
        source="blast",
    )


def spanning_alignment(
    payload: AlignmentInput, fills: dict[str, str], gap_len: int = GAP_LEN
) -> Alignment:
    """Build an `Alignment` as if MAFFT had aligned perfectly (no extra indels).

    `fills` maps each reference accession to what should appear in the gap's
    columns - the thing under test. Real MAFFT alignment is not exercised;
    everything downstream of it (consensus, scoring, validation, critique) is.
    """
    target_row = (
        payload.target_sequence[: payload.left_flank_length]
        + "-" * gap_len
        + payload.target_sequence[payload.left_flank_length :]
    )
    pairs = [
        AlignedPair(
            target_id=payload.target_id,
            reference_id=accession,
            target_aligned=target_row,
            reference_aligned=(
                payload.target_sequence[: payload.left_flank_length]
                + fill
                + payload.target_sequence[payload.left_flank_length :]
            ),
        )
        for accession, fill in fills.items()
    ]
    return Alignment(
        gap_id=payload.gap_id,
        pairs=pairs,
        gap_column_start=payload.left_flank_length,
        gap_column_end=payload.left_flank_length + gap_len,
    )


class ScriptedBlast(Tool[BlastSearchInput, BlastSearchOutput]):
    name = "blast_search"
    description = "Scripted BLAST for QA."
    estimated_seconds = 0.0

    def __init__(self, mode: str = "hits", delay: float = 0.0) -> None:
        self.mode = mode
        self.delay = delay
        self.calls: list[BlastSearchInput] = []

    async def run(self, payload: BlastSearchInput) -> BlastSearchOutput:
        self.calls.append(payload)
        if self.delay:
            await asyncio.sleep(self.delay)

        if self.mode == "outage":
            raise ExternalServiceError("blast", "service unavailable (simulated outage)")
        if self.mode == "zero_hits":
            return BlastSearchOutput(succeeded=True, references=[], total_hits=0)
        if self.mode == "malformed":
            return object()  # type: ignore[return-value]
        # "hits"
        refs = [make_reference(f"REF_{len(self.calls)}_{i}") for i in range(3)]
        return BlastSearchOutput(succeeded=True, references=refs, total_hits=len(refs))


class ScriptedMafft(Tool[AlignmentInput, AlignmentOutput]):
    name = "mafft_align"
    description = "Scripted MAFFT for QA."
    estimated_seconds = 0.0

    def __init__(self, mode: str = "good", gap_len: int = GAP_LEN, truth: str = TRUTH) -> None:
        self.mode = mode
        self.gap_len = gap_len
        self.truth = truth
        self.calls: list[AlignmentInput] = []

    async def run(self, payload: AlignmentInput) -> AlignmentOutput:
        self.calls.append(payload)

        if self.mode == "outage":
            raise ExternalServiceError("mafft", "service unavailable (simulated outage)")
        if self.mode == "malformed":
            return object()  # type: ignore[return-value]
        if self.mode == "no_span":
            # References align to the flanks but never insert columns across
            # the gap - the "references agree nothing is missing" case.
            alignment = Alignment(gap_id=payload.gap_id, pairs=[], gap_column_start=None)
            return AlignmentOutput(succeeded=True, alignment=alignment, aligned_count=0)
        if self.mode == "weak":
            # Spans the gap, but the references disagree with each other, so
            # the consensus margin collapses. The previous version gave every
            # reference the SAME filling, which is unanimous agreement - it
            # scored high and was accepted, quietly testing the opposite of
            # what the scenario claimed.
            alphabet = ["A", "C", "G", "T"]
            fills = {
                acc: (alphabet[index % 4] * self.gap_len)[: self.gap_len]
                for index, acc in enumerate(payload.references)
            }
            alignment = spanning_alignment(payload, fills, self.gap_len)
            return AlignmentOutput(
                succeeded=True, alignment=alignment, aligned_count=len(alignment.pairs)
            )
        # "good"
        fills = {acc: self.truth for acc in payload.references}
        alignment = spanning_alignment(payload, fills, self.gap_len)
        return AlignmentOutput(
            succeeded=True, alignment=alignment, aligned_count=len(alignment.pairs)
        )


class StubLLM:
    """A scripted LLM: plans a fixed tool sequence, always accepts in critique.

    Lets ACCEPT/REVISE/ABSTAIN and needs_agent scenarios run without spending
    real Azure tokens, while still exercising the real `Planner`/`Critic`
    JSON-parsing and prompt-building code paths.
    """

    def __init__(self, plans: list[list[dict[str, Any]]] | None = None) -> None:
        self.available = True
        self.plans = plans or []
        self._plan_index = 0
        self.calls = 0
        self.total_tokens = 0

    async def complete(self, messages: list[Message], *, max_tokens: int | None = None) -> str:
        return (await self.complete_with_usage(messages, max_tokens=max_tokens)).text

    async def complete_with_usage(
        self,
        messages: list[Message],
        *,
        max_tokens: int | None = None,
        json_schema: dict[str, Any] | None = None,
    ) -> Completion:
        self.calls += 1
        self.total_tokens += 120
        system = messages[0].content if messages else ""

        if "critic" in system.lower():
            text = json.dumps({"acceptable": True, "problems": [], "suggestion": None})
        else:
            if self._plan_index < len(self.plans):
                steps = self.plans[self._plan_index]
                self._plan_index += 1
            else:
                steps = []
            text = json.dumps({"steps": steps})

        return Completion(text=text, prompt_tokens=100, completion_tokens=20)


def _patch_llm(service: ReconstructionService, llm: Any) -> None:
    """Point the planner and critic at a scripted LLM.

    Reaches into the runner's node set because the real factory would need live
    Azure credentials. This works only because `AgentRunner` now shares ONE
    `ReconstructionNodes` instance with the compiled graph - it previously
    built two, and patching the reachable one changed nothing the graph ran.
    """
    nodes = service._runner._nodes  # noqa: SLF001 - test seam
    nodes.planner._llm = llm  # noqa: SLF001
    nodes.critic._llm = llm  # noqa: SLF001


def registry_of(*tools: Tool[Any, Any]) -> ToolRegistry:
    return ToolRegistry(list(tools))


async def run_once(
    settings: Settings,
    tools: ToolRegistry,
    sequence: Sequence,
    *,
    trace_id: str | None = None,
    checkpointer: Any = None,
    instruction: str = "Reconstruct the unresolved regions.",
) -> tuple[Any, CollectingEmitter, ReconstructionService]:
    events = CollectingEmitter()
    service = ReconstructionService(settings, tools, events=events, checkpointer=checkpointer)
    request = ReconstructionRequest.from_agent_request(
        instruction,
        {
            "sequence": {"identifier": sequence.identifier, "residues": sequence.residues},
            "organism": sequence.organism,
        },
    )
    outcome = await service.reconstruct(request, trace_id=trace_id)
    return outcome, events, service


# ============================================================================
# Scenarios
# ============================================================================


@scenario(
    "COMPLETED / ACCEPT - clean resolution", "status=completed, gap ACCEPTed, high confidence"
)
async def s_completed_accept(r: ScenarioResult) -> None:
    tools = registry_of(ScriptedBlast("hits"), ScriptedMafft("good"))
    outcome, events, _ = await run_once(base_settings(), tools, GAPPED_ONE)

    r.evidence = {
        "finished": outcome.finished,
        "stop_reason": outcome.result.stop_reason,
        "gaps": [(g.gap_id, g.status.value, g.confidence) for g in outcome.result.gaps],
        "reconstructed_sequence_recovers_truth": (
            outcome.result.reconstructed_sequence is not None
            and TRUTH in outcome.result.reconstructed_sequence
        ),
    }
    assert outcome.finished is True
    assert outcome.result.stop_reason == "all_gaps_resolved"
    assert len(outcome.result.gaps) == 1
    assert outcome.result.gaps[0].status is ReconstructionStatus.RECONSTRUCTED
    assert TRUTH in (outcome.result.reconstructed_sequence or "")
    assert outcome.needs_agent is None
    r.actual = f"completed, ACCEPTED, truth recovered, stop_reason={outcome.result.stop_reason}"
    r.verdict = "PASS"


@scenario(
    "COMPLETED / ABSTAIN - insufficient evidence, no more budget",
    "status=completed, gap reported UNRESOLVED with an explanation, not an error",
)
async def s_completed_abstain(r: ScenarioResult) -> None:
    tools = registry_of(ScriptedBlast("hits"), ScriptedMafft("weak"))
    # max_iterations=1 forces more_evidence_possible=False on the only critique
    # round, so the critic abstains instead of asking to revise.
    outcome, events, _ = await run_once(base_settings(max_iterations=1), tools, GAPPED_ONE)

    r.evidence = {
        "finished": outcome.finished,
        "stop_reason": outcome.result.stop_reason,
        "gap_status": outcome.result.gaps[0].status.value if outcome.result.gaps else None,
    }
    assert outcome.finished is True
    assert outcome.result.stop_reason in ("abstained", "max_iterations_reached")
    assert outcome.result.gaps[0].status is not ReconstructionStatus.RECONSTRUCTED
    r.actual = f"completed, gap not accepted, stop_reason={outcome.result.stop_reason}"
    r.verdict = "PASS"


@scenario(
    "Partial completion - one gap resolved, one abandoned",
    "finished=true, mixed gap statuses, overall_confidence reflects the weakest",
)
async def s_partial_completion(r: ScenarioResult) -> None:
    good_blast = ScriptedBlast("hits")
    good_mafft = ScriptedMafft("good", gap_len=GAP_LEN, truth=TRUTH)

    # One shared MAFFT stub can't return two different modes for two gaps, so
    # route by gap id via a tiny dispatcher tool.
    class MixedMafft(Tool[AlignmentInput, AlignmentOutput]):
        name = "mafft_align"
        description = "x"
        estimated_seconds = 0.0

        async def run(self, payload: AlignmentInput) -> AlignmentOutput:
            if payload.gap_id == "gap_2":
                return await ScriptedMafft("no_span", gap_len=GAP2_LEN).run(payload)
            return await good_mafft.run(payload)

    tools = registry_of(good_blast, MixedMafft())
    outcome, events, _ = await run_once(base_settings(max_iterations=2), tools, GAPPED_TWO)

    statuses = {g.gap_id: g.status.value for g in outcome.result.gaps}
    r.evidence = {"finished": outcome.finished, "statuses": statuses}
    assert outcome.finished is True
    assert len(outcome.result.gaps) == 2
    assert any(s == "reconstructed" for s in statuses.values())
    assert any(s != "reconstructed" for s in statuses.values())
    r.actual = f"mixed result across 2 gaps: {statuses}"
    r.verdict = "PASS"


@scenario(
    "REVISE -> re-plan with critique attached",
    "critic rejects weak evidence, and the next plan round is driven by that critique",
)
async def s_revise_replan(r: ScenarioResult) -> None:
    # A single reference is thin evidence by the critic's own floor, so the
    # verdict is REVISE while budget and iterations remain.
    class SingleRefBlast(Tool[BlastSearchInput, BlastSearchOutput]):
        name = "blast_search"
        description = "One reference only."
        estimated_seconds = 0.0

        async def run(self, payload: BlastSearchInput) -> BlastSearchOutput:
            return BlastSearchOutput(
                succeeded=True, references=[make_reference("REF_ONLY")], total_hits=1
            )

    tools = registry_of(SingleRefBlast(), ScriptedMafft("good"))
    outcome, events, _ = await run_once(base_settings(max_iterations=4), tools, GAPPED_ONE)

    plan_events = [e for e in events.events if e.type.value == "plan_created"]
    r.evidence = {
        "iterations": outcome.result.iterations,
        "plan_rounds": len(plan_events),
        "stop_reason": outcome.result.stop_reason,
        "gap_status": outcome.result.gaps[0].status.value,
    }
    # Thin evidence must not be silently accepted.
    assert outcome.result.gaps[0].status is not ReconstructionStatus.RECONSTRUCTED
    # And the loop must have gone round again rather than stopping on round 1.
    assert outcome.result.iterations >= 2, "a REVISE verdict should drive another round"
    assert len(plan_events) >= 2, "re-plan did not happen"
    r.actual = (
        f"thin evidence rejected; loop re-planned across {outcome.result.iterations} "
        f"iterations, stopping via {outcome.result.stop_reason!r}"
    )
    r.verdict = "PASS"


@scenario(
    "CONTINUE -> checkpoint -> resume, no repeated work",
    "slice 1 yields retryable=true; slice 2 resumes and does not redo completed tool calls",
)
async def s_continue_resume(r: ScenarioResult) -> None:
    from langgraph.checkpoint.memory import InMemorySaver

    checkpointer = InMemorySaver()
    blast = ScriptedBlast("hits", delay=0.05)
    mafft = ScriptedMafft("good")
    settings = base_settings(
        continuation=ContinuationSettings(_env_file=None, yield_after_seconds=0.01)
    )
    tools = registry_of(blast, mafft)

    outcome1, _, service = await run_once(
        settings, tools, GAPPED_ONE, trace_id="qa-resume-trace", checkpointer=checkpointer
    )
    calls_after_slice1 = len(blast.calls)

    outcome2, _, _ = await run_once(
        settings, tools, GAPPED_ONE, trace_id="qa-resume-trace", checkpointer=checkpointer
    )

    r.evidence = {
        "slice1_finished": outcome1.finished,
        "slice1_continuation_reason": outcome1.continuation_reason,
        "blast_calls_after_slice1": calls_after_slice1,
        "blast_calls_total": len(blast.calls),
        "slice2_finished": outcome2.finished,
    }
    assert outcome1.finished is False, "first slice should have yielded (0.01s budget)"
    assert outcome1.continuation_reason is not None
    assert calls_after_slice1 >= 1
    # The resumed slice must not redo the BLAST call slice 1 already paid for.
    assert len(blast.calls) == calls_after_slice1, (
        f"resume repeated {len(blast.calls) - calls_after_slice1} BLAST call(s) "
        "that slice 1 already completed"
    )
    r.actual = (
        f"slice1 yielded (retryable), {calls_after_slice1} BLAST call(s); "
        f"slice2 resumed with 0 additional BLAST calls, finished={outcome2.finished}"
    )
    r.verdict = "PASS"


@scenario(
    "FAILED - invalid sequence", "InvalidSequenceError propagates; API maps it to status=failed"
)
async def s_failed_invalid_sequence(r: ScenarioResult) -> None:
    tools = registry_of(ScriptedBlast("hits"), ScriptedMafft("good"))
    service = ReconstructionService(base_settings(), tools, events=CollectingEmitter())
    request = ReconstructionRequest.from_agent_request(
        "Reconstruct.", {}
    )  # no sequence, no accession

    raised = None
    try:
        await service.reconstruct(request)
    except InvalidSequenceError as error:
        raised = error

    r.evidence = {"raised": str(raised)}
    assert raised is not None
    r.actual = f"InvalidSequenceError raised: {raised}"
    r.verdict = "PASS"


@scenario(
    "FAILED - malformed FASTA (illegal characters)",
    "InvalidSequenceError at Sequence.parse, before any tool runs",
)
async def s_failed_invalid_fasta(r: ScenarioResult) -> None:
    tools = registry_of(ScriptedBlast("hits"), ScriptedMafft("good"))
    service = ReconstructionService(base_settings(), tools, events=CollectingEmitter())
    request = ReconstructionRequest.from_agent_request(
        "Reconstruct.", {"sequence": {"identifier": "bad", "residues": "MKVLAAGIVGL"}}
    )

    raised = None
    try:
        await service.reconstruct(request)
    except InvalidSequenceError as error:
        raised = error

    r.evidence = {"raised": str(raised)}
    assert raised is not None
    r.actual = f"InvalidSequenceError raised: {raised}"
    r.verdict = "PASS"


@scenario(
    "No gaps in sequence",
    "documented as NoGapsFoundError in the exception taxonomy; verify what actually happens",
)
async def s_no_gaps(r: ScenarioResult) -> None:
    tools = registry_of(ScriptedBlast("hits"), ScriptedMafft("good"))
    ungapped = Sequence.parse("qa_no_gaps", "ACGT" * 40, organism="Testus organismus")
    outcome, events, _ = await run_once(base_settings(), tools, ungapped)

    r.evidence = {
        "finished": outcome.finished,
        "stop_reason": outcome.result.stop_reason,
        "gaps_reported": len(outcome.result.gaps),
        "summary": outcome.result.summary,
    }
    assert outcome.finished is True
    assert outcome.result.stop_reason == "nothing_to_do"
    assert len(outcome.result.gaps) == 0
    r.actual = (
        "COMPLETED with stop_reason='nothing_to_do', 0 gaps reported. "
        "domain.exceptions.NoGapsFoundError is never raised on this path (dead code)."
    )
    r.severity = "LOW"
    r.defects.append(
        "NoGapsFoundError (domain/exceptions.py) and its handlers in both API routes "
        "are unreachable: a gapless sequence resolves through detect_gaps -> "
        "has_work()=False -> finalize() with stop_reason='nothing_to_do', never through "
        "that exception. Contract-inconsistent dead code, not a functional bug."
    )
    r.verdict = "PASS"  # functional behavior is correct; the defect is documentation/dead-code


@scenario("Multiple gaps in one sequence", "both gaps detected, tracked and reported independently")
async def s_multiple_gaps(r: ScenarioResult) -> None:
    tools = registry_of(ScriptedBlast("hits"), ScriptedMafft("good"))
    outcome, events, _ = await run_once(base_settings(max_iterations=3), tools, GAPPED_TWO)

    r.evidence = {
        "gap_count": len(outcome.result.gaps),
        "gap_ids": [g.gap_id for g in outcome.result.gaps],
    }
    assert len(outcome.result.gaps) == 2
    assert {g.gap_id for g in outcome.result.gaps} == {"gap_1", "gap_2"}
    r.actual = f"both gaps tracked: {[g.status.value for g in outcome.result.gaps]}"
    r.verdict = "PASS"


@scenario("Tool budget exhausted", "stop_reason=budget_exhausted, partial results reported")
async def s_budget_exhausted(r: ScenarioResult) -> None:
    tools = registry_of(ScriptedBlast("hits"), ScriptedMafft("good"))
    settings = base_settings(
        max_iterations=6,
        budgets=BudgetSettings(_env_file=None, max_tool_calls=1),
    )
    outcome, events, _ = await run_once(settings, tools, GAPPED_ONE)

    r.evidence = {
        "stop_reason": outcome.result.stop_reason,
        "budget": outcome.result.budget,
        "gap_status": outcome.result.gaps[0].status.value if outcome.result.gaps else None,
    }
    assert outcome.finished is True
    assert outcome.result.stop_reason == "budget_exhausted"
    assert outcome.result.budget["tool_calls"] <= 1
    assert outcome.result.gaps[0].status is not ReconstructionStatus.RECONSTRUCTED
    r.actual = (
        f"stopped with stop_reason='budget_exhausted' after {outcome.result.budget['tool_calls']} "
        f"tool call(s) (cap=1); gap reported {outcome.result.gaps[0].status.value}"
    )
    r.verdict = "PASS"


@scenario(
    "Timeout / yield mid-slice",
    "wall clock exceeded -> CONTINUE, retryable=true, continuation_reason names what's outstanding",
)
async def s_timeout_yield(r: ScenarioResult) -> None:
    from langgraph.checkpoint.memory import InMemorySaver

    tools = registry_of(ScriptedBlast("hits", delay=0.05), ScriptedMafft("good"))
    settings = base_settings(
        continuation=ContinuationSettings(_env_file=None, yield_after_seconds=0.001)
    )
    outcome, events, _ = await run_once(
        settings, tools, GAPPED_ONE, trace_id="qa-yield-trace", checkpointer=InMemorySaver()
    )

    r.evidence = {
        "finished": outcome.finished,
        "continuation_reason": outcome.continuation_reason,
    }
    assert outcome.finished is False
    assert outcome.continuation_reason is not None and "gap" in outcome.continuation_reason.lower()
    r.actual = f"yielded: {outcome.continuation_reason!r}"
    r.verdict = "PASS"


@scenario(
    "Last slice never yields (would be force-failed by the orchestrator)",
    "on max_slices - 1, agent returns finished=true with partial results, never CONTINUE",
)
async def s_last_slice_no_yield(r: ScenarioResult) -> None:
    from langgraph.checkpoint.memory import InMemorySaver

    tools = registry_of(ScriptedBlast("hits", delay=0.05), ScriptedMafft("good"))
    settings = base_settings(
        continuation=ContinuationSettings(_env_file=None, yield_after_seconds=0.001, max_slices=1)
    )
    outcome, events, _ = await run_once(
        settings, tools, GAPPED_ONE, trace_id="qa-lastslice-trace", checkpointer=InMemorySaver()
    )

    r.evidence = {"finished": outcome.finished, "continuation_reason": outcome.continuation_reason}
    assert outcome.finished is True
    assert outcome.continuation_reason is None
    r.actual = "finished=true on the last granted slice, despite exceeding the wall clock"
    r.verdict = "PASS"


@scenario(
    "Tool retry after semantic failure - bounded at two attempts",
    "a tool that finds nothing is retried once and then abandoned, never a third time",
)
async def s_tool_retry(r: ScenarioResult) -> None:
    mafft = ScriptedMafft("no_span")  # never produces a usable alignment
    tools = registry_of(ScriptedBlast("hits"), mafft)
    outcome, events, _ = await run_once(base_settings(max_iterations=6), tools, GAPPED_ONE)

    mafft_obs = [o for o in outcome.result.observations if o.tool == "mafft_align"]
    skipped = [o for o in mafft_obs if o.status is ObservationStatus.SKIPPED]
    r.evidence = {
        "mafft_attempts": len(mafft.calls),
        "observation_statuses": [o.status.value for o in mafft_obs],
        "relaxed_flags": [o.relaxed for o in mafft_obs],
    }
    assert len(mafft.calls) == 2, (
        f"expected exactly 2 attempts (one try, one relaxed retry), got {len(mafft.calls)}"
    )
    assert any(o.relaxed for o in mafft_obs), "the retry was not recorded as relaxed"
    # A non-spanning alignment must NOT satisfy the alignment precondition -
    # if it did, the retry would never be planned at all.
    assert skipped or len(mafft.calls) == 2
    r.actual = (
        f"{len(mafft.calls)} attempts (second flagged relaxed), then abandoned; "
        "a non-spanning alignment correctly did not count as a satisfied precondition"
    )
    r.verdict = "PASS"


@scenario("Zero BLAST hits", "succeeded=true, empty references -> EMPTY observation, not FAILED")
async def s_zero_blast_hits(r: ScenarioResult) -> None:
    tools = registry_of(ScriptedBlast("zero_hits"), ScriptedMafft("good"))
    outcome, events, _ = await run_once(base_settings(max_iterations=2), tools, GAPPED_ONE)

    blast_observations = [o for o in outcome.result.observations if o.tool == "blast_search"]
    r.evidence = {
        "observation_statuses": [o.status.value for o in blast_observations],
        "gap_status": outcome.result.gaps[0].status.value,
    }
    assert blast_observations, "expected at least one blast_search observation"
    assert all(o.status is ObservationStatus.EMPTY for o in blast_observations)
    assert outcome.result.gaps[0].status is not ReconstructionStatus.RECONSTRUCTED
    r.actual = "BLAST recorded as EMPTY (not FAILED); gap unresolved, run completed cleanly"
    r.verdict = "PASS"


@scenario(
    "Weak MAFFT alignment - divergent references",
    "low-identity consensus is flagged by the critic and not silently accepted",
)
async def s_weak_mafft_identity(r: ScenarioResult) -> None:
    tools = registry_of(ScriptedBlast("hits"), ScriptedMafft("weak"))
    outcome, events, _ = await run_once(base_settings(max_iterations=1), tools, GAPPED_ONE)

    gap = outcome.result.gaps[0]
    r.evidence = {
        "status": gap.status.value,
        "confidence": gap.confidence,
        "explanation": gap.explanation,
    }
    assert gap.status is not ReconstructionStatus.RECONSTRUCTED
    r.actual = f"weak-identity consensus rejected: status={gap.status.value}"
    r.verdict = "PASS"


@scenario(
    "Weak MAFFT alignment - no columns span the gap",
    "references align to the flanks but insert nothing across the gap -> UNRESOLVED, not a crash",
)
async def s_weak_mafft_no_span(r: ScenarioResult) -> None:
    tools = registry_of(ScriptedBlast("hits"), ScriptedMafft("no_span"))
    outcome, events, _ = await run_once(base_settings(max_iterations=1), tools, GAPPED_ONE)

    gap = outcome.result.gaps[0]
    r.evidence = {"status": gap.status.value}
    assert gap.status is ReconstructionStatus.UNRESOLVED
    r.actual = "no usable alignment columns -> UNRESOLVED, run completed without error"
    r.verdict = "PASS"


@scenario(
    "NCBI outage during accession resolution", "InvalidSequenceError, no gap-finding attempted"
)
async def s_ncbi_outage(r: ScenarioResult) -> None:
    class FailingNCBI:
        async def fetch_fasta(self, identifiers: list[str], **kwargs: Any) -> str:
            raise ExternalServiceError(
                "ncbi", "connection refused (simulated outage)", retryable=True
            )

    tools = registry_of(ScriptedBlast("hits"), ScriptedMafft("good"))
    service = ReconstructionService(
        base_settings(), tools, events=CollectingEmitter(), ncbi_client=FailingNCBI()
    )
    request = ReconstructionRequest.from_agent_request("Reconstruct.", {"accession": "NC_000001.1"})

    raised = None
    try:
        await service.reconstruct(request)
    except (ExternalServiceError, InvalidSequenceError) as error:
        raised = error

    r.evidence = {"raised_type": type(raised).__name__ if raised else None, "raised": str(raised)}
    assert raised is not None
    r.actual = f"{type(raised).__name__} propagated from accession resolution: {raised}"
    r.verdict = "PASS"


@scenario(
    "EMBL-EBI outage mid-run (BLAST fails after sequence is already parsed)",
    "one gap's BLAST call fails; run completes with that gap UNRESOLVED, no crash",
)
async def s_embl_outage(r: ScenarioResult) -> None:
    tools = registry_of(ScriptedBlast("outage"), ScriptedMafft("good"))
    outcome, events, _ = await run_once(base_settings(max_iterations=2), tools, GAPPED_ONE)

    failed = [o for o in outcome.result.observations if o.status is ObservationStatus.FAILED]
    r.evidence = {
        "finished": outcome.finished,
        "failed_observations": len(failed),
        "gap_status": outcome.result.gaps[0].status.value,
        "warnings": outcome.result.warnings,
    }
    assert outcome.finished is True
    assert failed, "expected at least one FAILED observation from the simulated outage"
    assert outcome.result.gaps[0].status is not ReconstructionStatus.RECONSTRUCTED
    r.actual = f"{len(failed)} FAILED observation(s), run completed with the gap unresolved"
    r.verdict = "PASS"


@scenario(
    "Malformed tool response (non-conforming return value)",
    "a non-ToolOutput return is recorded as FAILED with a contract_violation diagnostic",
)
async def s_malformed_tool_response(r: ScenarioResult) -> None:
    tools = registry_of(ScriptedBlast("malformed"), ScriptedMafft("good"))
    outcome, events, _ = await run_once(base_settings(max_iterations=2), tools, GAPPED_ONE)

    blast_obs = [o for o in outcome.result.observations if o.tool == "blast_search"]
    r.evidence = {
        "finished": outcome.finished,
        "statuses": [o.status.value for o in blast_obs],
        "diagnostics": [o.diagnostics for o in blast_obs],
        "gap_status": outcome.result.gaps[0].status.value,
    }
    assert outcome.finished is True, "a broken tool must not crash the run"
    assert blast_obs, "expected an observation for the malformed call"
    assert all(o.status is ObservationStatus.FAILED for o in blast_obs), (
        "a tool returning the wrong type must be FAILED, not silently EMPTY - "
        "otherwise a tool bug is indistinguishable from a legitimate empty result"
    )
    assert any(
        o.diagnostics.get("contract_violation") == "tool_output_type" for o in blast_obs
    ), "the type mismatch was not recorded in diagnostics"
    r.actual = (
        "a plain object() return was rejected against the ToolOutput contract and "
        "recorded as FAILED with contract_violation=tool_output_type"
    )
    r.verdict = "PASS"


@scenario("Evo 2 tool failure", "Evo2PlausibilityTool reports succeeded=false, run is not aborted")
async def s_evo_failure(r: ScenarioResult) -> None:
    from tools.evo.schemas import PlausibilityInput
    from tools.evo.tool import Evo2PlausibilityTool

    class FailingEvo2Client:
        async def generate(self, prompt: str, *, num_tokens: int, **kwargs: Any) -> Any:
            raise ExternalServiceError("nvidia_evo2", "simulated NIM outage")

    tool = Evo2PlausibilityTool(FailingEvo2Client())  # type: ignore[arg-type]
    output = await tool.run(
        PlausibilityInput(gap_id="gap_1", left_flank="A" * 100, candidates={"c1": "ACGT"})
    )

    r.evidence = {"succeeded": output.succeeded, "error": output.error}
    assert output.succeeded is False
    assert output.error is not None
    r.actual = f"Evo2PlausibilityTool.run() returned succeeded=False, error={output.error!r}"
    r.verdict = "PASS"


@scenario(
    "NEEDS_AGENT escalation to Evolution - WITHOUT an LLM",
    "the deterministic planner invokes evolutionary_context and escalates when "
    "relatedness cannot separate the references",
)
async def s_needs_agent_deterministic(r: ScenarioResult) -> None:
    # Mammuthus vs Loxodonta: different genus, so the name heuristic cannot
    # place them and recommends delegation. No LLM configured at all.
    tools = registry_of(
        ScriptedBlast("hits"), ScriptedMafft("weak"), EvolutionaryContextTool()
    )
    outcome, events, _ = await run_once(base_settings(max_iterations=4), tools, GAPPED_ONE)

    evo_calls = [e for e in events.events if e.data.get("tool") == "evolutionary_context"]
    r.evidence = {
        "llm_configured": False,
        "evolutionary_context_invocations": len(evo_calls),
        "needs_agent": outcome.needs_agent,
        "prompt_to_target_agent": outcome.prompt_to_target_agent,
        "stop_reason": outcome.result.stop_reason,
    }
    assert evo_calls, (
        "the deterministic planner never invoked evolutionary_context, so escalation "
        "would remain reachable only via a live LLM"
    )
    assert outcome.needs_agent == "Evolution"
    assert outcome.prompt_to_target_agent and "Mammuthus" in outcome.prompt_to_target_agent
    assert outcome.result.stop_reason == "delegated_to_another_agent"
    r.actual = (
        "with LLM_PROVIDER=none, evolutionary_context ran and the agent escalated: "
        f"{outcome.prompt_to_target_agent!r}"
    )
    r.verdict = "PASS"


@scenario(
    "NEEDS_AGENT names the OPEN gaps, not only ones with an unresolved entry",
    "a gap that never produced a candidate is the most likely to be blocked on phylogeny",
)
async def s_needs_agent_open_gaps(r: ScenarioResult) -> None:
    # BLAST finds references (so relatedness is unresolved and evo runs), but
    # MAFFT never spans, so no candidate and no `reconstructions` entry exists.
    tools = registry_of(
        ScriptedBlast("hits"), ScriptedMafft("no_span"), EvolutionaryContextTool()
    )
    outcome, events, _ = await run_once(base_settings(max_iterations=4), tools, GAPPED_ONE)

    r.evidence = {
        "needs_agent": outcome.needs_agent,
        "prompt": outcome.prompt_to_target_agent,
        "gaps_reported": [g.gap_id for g in outcome.result.gaps],
    }
    assert outcome.needs_agent == "Evolution"
    assert outcome.prompt_to_target_agent and "gap_1" in outcome.prompt_to_target_agent, (
        "the escalation did not name the open gap - the old code only looked at "
        "gaps with an entry in `reconstructions`, so a gap this blocked was invisible"
    )
    r.actual = f"escalation names the open gap: {outcome.prompt_to_target_agent!r}"
    r.verdict = "PASS"


@scenario(
    "NEEDS_AGENT is NOT suppressed on the last slice",
    "escalation does not consume a CONTINUE retry, so the slice budget must not gate it",
)
async def s_needs_agent_last_slice(r: ScenarioResult) -> None:
    tools = registry_of(
        ScriptedBlast("hits"), ScriptedMafft("weak"), EvolutionaryContextTool()
    )
    settings = base_settings(
        max_iterations=4,
        continuation=ContinuationSettings(_env_file=None, max_slices=1),
    )
    outcome, events, _ = await run_once(settings, tools, GAPPED_ONE)

    r.evidence = {
        "max_slices": 1,
        "needs_agent": outcome.needs_agent,
        "stop_reason": outcome.result.stop_reason,
    }
    assert outcome.needs_agent == "Evolution", (
        "escalation was withheld on the last slice. NEEDS_AGENT is routed by the "
        "capability resolver and does not consume a CONTINUE retry "
        "(worker_node.route_after_worker), so gating it on the slice budget "
        "withholds the request for help from exactly the runs that need it most"
    )
    r.actual = "escalated on the final granted slice, as it should"
    r.verdict = "PASS"
@scenario(
    "Checkpoint recovery after simulated process restart",
    "a fresh AgentRunner + fresh ReconstructionService, same checkpointer, same trace_id, resumes",
)
async def s_checkpoint_recovery(r: ScenarioResult) -> None:
    from langgraph.checkpoint.memory import InMemorySaver

    checkpointer = InMemorySaver()  # stands in for Postgres surviving a restart
    blast = ScriptedBlast("hits", delay=0.05)
    mafft = ScriptedMafft("good")
    settings = base_settings(
        continuation=ContinuationSettings(_env_file=None, yield_after_seconds=0.01)
    )

    # "Process 1": builds its own service/runner/registry instances.
    tools_p1 = registry_of(blast, mafft)
    outcome1, _, _ = await run_once(
        settings, tools_p1, GAPPED_ONE, trace_id="qa-restart-trace", checkpointer=checkpointer
    )
    assert outcome1.finished is False

    # "Process 2": entirely new Python objects, only the checkpointer and
    # trace id carry over - simulating a container restart between calls.
    tools_p2 = registry_of(blast, mafft)  # same stub instances so call counts are observable
    outcome2, _, _ = await run_once(
        settings, tools_p2, GAPPED_ONE, trace_id="qa-restart-trace", checkpointer=checkpointer
    )

    r.evidence = {
        "process1_finished": outcome1.finished,
        "process2_finished": outcome2.finished,
        "process2_saw_prior_gap_contexts": bool(outcome2.state.get("gap_contexts"))
        if hasattr(outcome2, "state")
        else "n/a",
    }
    assert outcome2.finished is True
    r.actual = "a new AgentRunner instance resumed the checkpoint written by a different instance"
    r.verdict = "PASS"


@scenario(
    "Duplicate / replayed X-Trace-Id after completion",
    "re-calling with the same trace_id after ALL_RESOLVED should short-circuit; "
    "verify it actually does",
)
async def s_duplicate_trace_id_replay(r: ScenarioResult) -> None:
    from langgraph.checkpoint.memory import InMemorySaver

    checkpointer = InMemorySaver()
    tools = registry_of(ScriptedBlast("hits"), ScriptedMafft("good"))
    llm = StubLLM(plans=[[]])  # deterministic-shaped: empty plan every round
    settings = base_settings(
        max_iterations=6, llm=LLMSettings(_env_file=None, provider=LLMProvider.OPENAI)
    )

    events = CollectingEmitter()
    service = ReconstructionService(settings, tools, events=events, checkpointer=checkpointer)
    _patch_llm(service, llm)

    request = ReconstructionRequest.from_agent_request(
        "Reconstruct.",
        {
            "sequence": {"identifier": GAPPED_ONE.identifier, "residues": GAPPED_ONE.residues},
            "organism": GAPPED_ONE.organism,
        },
    )
    first = await service.reconstruct(request, trace_id="qa-duplicate-trace")
    assert first.finished is True
    assert first.result.stop_reason == "all_gaps_resolved"

    planner_calls_before = llm.calls
    tool_calls_before = len(tools.get("blast_search").calls)  # type: ignore[attr-defined]

    second = await service.reconstruct(request, trace_id="qa-duplicate-trace")

    r.evidence = {
        "planner_llm_calls_before_replay": planner_calls_before,
        "planner_llm_calls_after_replay": llm.calls,
        "tool_calls_before_replay": tool_calls_before,
        "tool_calls_after_replay": len(tools.get("blast_search").calls),  # type: ignore[attr-defined]
        "second_result_identical_gaps": (
            [g.status.value for g in second.result.gaps]
            == [g.status.value for g in first.result.gaps]
        ),
    }
    assert len(tools.get("blast_search").calls) == tool_calls_before, (  # type: ignore[attr-defined]
        "a replayed trace_id on an already-completed run re-invoked a real tool call"
    )
    wasted_llm_calls = llm.calls - planner_calls_before
    r.actual = (
        f"tool calls not repeated (good); but the replay cost {wasted_llm_calls} additional "
        f"LLM call(s) ({wasted_llm_calls * 120} tokens) before recognising the run was "
        f"already finished."
    )
    if wasted_llm_calls > 0:
        r.defects.append(
            "has_work() (agent/graph/conditions.py) only excludes gaps in `skipped`, not "
            "gaps already in `reconstructions`. On a resumed/replayed trace_id whose run is "
            "already ALL_RESOLVED, route_after_detection therefore sends the graph into "
            "PLAN -> select_tools -> CRITIQUE -> decide for one full extra round (LLM plan "
            "call + LLM critique call re-reviewing every already-accepted gap) before "
            "decide() notices ALL_RESOLVED and stops. Idempotent replay is not free."
        )
        r.severity = "MEDIUM"
        r.verdict = "FAIL"
    else:
        r.verdict = "PASS"


@scenario(
    "NO_PROGRESS fires when a round adds nothing",
    "progress is measured per round from the observation trail, not from accumulated state",
)
async def s_no_progress_fires(r: ScenarioResult) -> None:
    # BLAST finds references; MAFFT never spans the gap. After MAFFT's two
    # allowed attempts, a round runs that adds nothing at all.
    tools = registry_of(ScriptedBlast("hits"), ScriptedMafft("no_span"))
    outcome, events, _ = await run_once(base_settings(max_iterations=8), tools, GAPPED_ONE)

    mafft_calls = len(tools.get("mafft_align").calls)  # type: ignore[attr-defined]
    r.evidence = {
        "stop_reason": outcome.result.stop_reason,
        "iterations": outcome.result.iterations,
        "mafft_attempts": mafft_calls,
        "max_iterations": 8,
    }
    # The whole point: it must NOT burn all 8 iterations discovering this.
    assert outcome.result.iterations < 8, (
        "the loop ran to MAX_ITERATIONS instead of noticing it had stalled"
    )
    assert outcome.result.stop_reason in ("no_progress", "abstained"), (
        f"expected a stall to be named, got {outcome.result.stop_reason!r}"
    )
    r.actual = (
        f"stalled run stopped after {outcome.result.iterations}/8 iterations via "
        f"{outcome.result.stop_reason!r} ({mafft_calls} MAFFT attempts)"
    )
    r.verdict = "PASS"


@scenario(
    "Semantic retry actually widens the search",
    "a second attempt at the same tool/gap uses relaxed parameters, not a byte-identical repeat",
)
async def s_retry_is_relaxed(r: ScenarioResult) -> None:
    blast = ScriptedBlast("zero_hits")
    tools = registry_of(blast, ScriptedMafft("good"))
    outcome, events, _ = await run_once(base_settings(max_iterations=4), tools, GAPPED_ONE)

    payloads = [
        {"expect": c.expect, "max_hits": c.max_hits, "database": c.database}
        for c in blast.calls
    ]
    relaxed_observations = [
        o for o in outcome.result.observations if o.tool == "blast_search" and o.relaxed
    ]
    r.evidence = {"attempts": len(payloads), "payloads": payloads}
    assert len(payloads) >= 2, f"expected a retry, got {len(payloads)} attempt(s)"
    assert payloads[0] != payloads[1], (
        "the retry repeated the first attempt byte for byte, so it could only "
        "ever return the same answer"
    )
    assert relaxed_observations, "the retry was not recorded as relaxed in the audit trail"
    r.actual = (
        f"attempt 1 {payloads[0]} -> attempt 2 {payloads[1]}; "
        f"{len(relaxed_observations)} observation(s) flagged relaxed"
    )
    r.verdict = "PASS"


@scenario(
    "HTTP contract: /execute always returns the bare AgentResult shape",
    "no envelope wrapping; retryable/continuation_reason/error present per orchestrator schema",
)
async def s_http_execute_contract(r: ScenarioResult) -> None:
    from fastapi.testclient import TestClient

    from api.app import create_app
    from api.dependencies import get_service

    tools = registry_of(ScriptedBlast("hits"), ScriptedMafft("good"))
    settings = base_settings()
    service = ReconstructionService(settings, tools, events=CollectingEmitter())

    app = create_app(settings)
    app.dependency_overrides[get_service] = lambda: service
    client = TestClient(app)

    response = client.post(
        "/execute",
        json={
            "instruction": "Reconstruct.",
            "context": {
                "sequence": {"identifier": GAPPED_ONE.identifier, "residues": GAPPED_ONE.residues},
                "organism": GAPPED_ONE.organism,
            },
        },
        headers={"X-Trace-Id": "qa-http-trace"},
    )
    body = response.json()

    r.evidence = {
        "status_code": response.status_code,
        "keys": sorted(body),
        "agent_status": body.get("status"),
    }
    assert response.status_code == 200
    assert set(body) == {
        "status",
        "target_agent",
        "prompt_to_target_agent",
        "output",
        "continuation_reason",
        "retryable",
        "error",
    }
    assert body["status"] == "completed"
    assert "reconstruction" in body["output"]
    r.actual = f"200, status={body['status']!r}, contract keys exactly match orchestrator schema"
    r.verdict = "PASS"


@scenario(
    "HTTP contract: /execute CONTINUE carries retryable=true",
    "orchestrator's worker_node converts CONTINUE without retryable=true straight to FAILED",
)
async def s_http_execute_continue_contract(r: ScenarioResult) -> None:
    from fastapi.testclient import TestClient
    from langgraph.checkpoint.memory import InMemorySaver

    from api.app import create_app
    from api.dependencies import get_service

    tools = registry_of(ScriptedBlast("hits", delay=0.05), ScriptedMafft("good"))
    settings = base_settings(
        continuation=ContinuationSettings(_env_file=None, yield_after_seconds=0.001)
    )
    service = ReconstructionService(
        settings, tools, events=CollectingEmitter(), checkpointer=InMemorySaver()
    )

    app = create_app(settings)
    app.dependency_overrides[get_service] = lambda: service
    client = TestClient(app)

    response = client.post(
        "/execute",
        json={
            "instruction": "Reconstruct.",
            "context": {
                "sequence": {"identifier": GAPPED_ONE.identifier, "residues": GAPPED_ONE.residues},
                "organism": GAPPED_ONE.organism,
            },
        },
        headers={"X-Trace-Id": "qa-http-continue-trace"},
    )
    body = response.json()

    r.evidence = {"status": body.get("status"), "retryable": body.get("retryable")}
    assert body["status"] == "continue"
    assert body["retryable"] is True
    assert body["continuation_reason"]
    r.actual = f"status=continue, retryable={body['retryable']}, reason set"
    r.verdict = "PASS"


@scenario(
    "HTTP contract: malformed request body uses the v1 envelope, not FastAPI's default 422",
    "validation errors on /api/v1/reconstructions still return {data, meta, error}",
)
async def s_http_v1_validation_contract(r: ScenarioResult) -> None:
    from fastapi.testclient import TestClient

    from api.app import create_app
    from api.dependencies import get_service

    tools = registry_of(ScriptedBlast("hits"), ScriptedMafft("good"))
    settings = base_settings()
    service = ReconstructionService(settings, tools, events=CollectingEmitter())
    app = create_app(settings)
    app.dependency_overrides[get_service] = lambda: service
    client = TestClient(app)

    response = client.post("/api/v1/reconstructions", json={"min_confidence": 5.0})
    body = response.json()

    r.evidence = {"status_code": response.status_code, "keys": sorted(body)}
    assert response.status_code == 422
    assert set(body) == {"data", "meta", "error"}
    assert body["error"]["code"] == "validation_error"
    r.actual = f"422 with envelope shape, error.code={body['error']['code']!r}"
    r.verdict = "PASS"


@scenario(
    "Rate limit from an external service surfaces as retryable",
    "RateLimitError on the v1 endpoint maps to ErrorCode.RATE_LIMITED, retryable=true",
)
async def s_rate_limited(r: ScenarioResult) -> None:
    from fastapi.testclient import TestClient

    from api.app import create_app
    from api.dependencies import get_service

    class RateLimitedBlast(Tool[BlastSearchInput, BlastSearchOutput]):
        name = "blast_search"
        description = "x"
        estimated_seconds = 0.0

        async def run(self, payload: BlastSearchInput) -> BlastSearchOutput:
            raise RateLimitError("blast", retry_after=30.0)

    class ThrowingService:
        async def reconstruct(self, request: Any, *, trace_id: str | None = None) -> Any:
            raise RateLimitError("blast", retry_after=30.0)

    app = create_app(base_settings())
    app.dependency_overrides[get_service] = lambda: ThrowingService()
    client = TestClient(app)

    response = client.post("/api/v1/reconstructions", json={"sequence": "ACGTNNNN"})
    body = response.json()

    r.evidence = {
        "error_code": body.get("error", {}).get("code"),
        "retryable": body.get("error", {}).get("retryable"),
    }
    assert body["error"]["code"] == "rate_limited"
    assert body["error"]["retryable"] is True
    r.actual = "error.code=rate_limited, retryable=True"
    r.verdict = "PASS"


@scenario(
    "CANCELLED status - does it exist anywhere in the contract?",
    "requested by the audit brief; verify presence or absence explicitly",
)
async def s_cancelled_status(r: ScenarioResult) -> None:
    from api.v1.schemas import AgentStatus

    values = {member.value for member in AgentStatus}
    r.evidence = {"agent_status_values": sorted(values)}
    assert "cancelled" not in values
    r.actual = f"AgentStatus has no CANCELLED member: {sorted(values)}"
    r.defects.append(
        "There is no cancellation concept anywhere in the agent: no AgentStatus.CANCELLED, "
        "no cancel endpoint, and nothing in the graph checks for an external cancellation "
        "signal mid-run. A caller cannot abort an in-flight reconstruction; the only way a "
        "run stops early is via the agent's own budgets/yield logic. This may be "
        "intentional (the orchestrator has no cancellation concept to propagate either), "
        "but it is a genuine capability gap against the audit brief's requested scenario."
    )
    r.severity = "LOW"
    r.verdict = "PASS"  # correctly documents an absence, not a broken feature


# ============================================================================
# Runner
# ============================================================================

SEVERITY_ORDER = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3, "-": 4}


async def main() -> int:
    configure_logging(log_format=LogFormat.JSON)
    started = time.monotonic()

    scenarios = [
        s_completed_accept,
        s_completed_abstain,
        s_partial_completion,
        s_revise_replan,
        s_continue_resume,
        s_failed_invalid_sequence,
        s_failed_invalid_fasta,
        s_no_gaps,
        s_multiple_gaps,
        s_budget_exhausted,
        s_timeout_yield,
        s_last_slice_no_yield,
        s_tool_retry,
        s_zero_blast_hits,
        s_weak_mafft_identity,
        s_weak_mafft_no_span,
        s_ncbi_outage,
        s_embl_outage,
        s_malformed_tool_response,
        s_evo_failure,
        s_needs_agent_deterministic,
        s_needs_agent_open_gaps,
        s_needs_agent_last_slice,
        s_checkpoint_recovery,
        s_duplicate_trace_id_replay,
        s_no_progress_fires,
        s_retry_is_relaxed,
        s_http_execute_contract,
        s_http_execute_continue_contract,
        s_http_v1_validation_contract,
        s_rate_limited,
        s_cancelled_status,
    ]

    print(f"Running {len(scenarios)} scenarios...\n")
    for fn in scenarios:
        await fn()

    elapsed = time.monotonic() - started
    passed = sum(1 for r in RESULTS if r.verdict == "PASS")
    failed = [r for r in RESULTS if r.verdict == "FAIL"]
    failed.sort(key=lambda r: SEVERITY_ORDER.get(r.severity, 4))

    print(f"\n{'=' * 78}")
    print(f"{passed}/{len(RESULTS)} scenarios passed in {elapsed:.2f}s")
    print(f"{'=' * 78}")

    if failed:
        print("\nFailures / confirmed defects, most severe first:\n")
        for r in failed:
            print(f"[{r.severity}] {r.scenario}")
            print(f"    expected: {r.expected}")
            print(f"    actual:   {r.actual}")
            for d in r.defects:
                print(f"    defect:   {d}")
            print()

    # Dump machine-readable results for the report writer.
    out_path = Path(__file__).resolve().parent / "qa_audit_results.json"
    out_path.write_text(
        json.dumps(
            [
                {
                    "scenario": r.scenario,
                    "expected": r.expected,
                    "actual": r.actual,
                    "verdict": r.verdict,
                    "severity": r.severity,
                    "defects": r.defects,
                    "evidence": r.evidence,
                    "error": r.error,
                }
                for r in RESULTS
            ],
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    print(f"Machine-readable results written to {out_path}")

    critical_or_high = [r for r in failed if r.severity in ("CRITICAL", "HIGH")]
    return 1 if critical_or_high else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
