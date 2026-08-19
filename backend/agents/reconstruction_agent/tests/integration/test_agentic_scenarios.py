"""Every agentic scenario and status transition, as tracked CI.

This is the QA audit in `scripts/qa_audit.py` promoted into the test suite. The
script stays useful as a one-shot report with severities and a readiness
verdict; this file is what stops a regression from ever reaching a branch.

Both drive the same production code - `ReconstructionService`, `AgentRunner`,
the compiled LangGraph loop, the stop policy, the critic, the budget policy and
the FastAPI routes. Only the network edges are replaced, by stand-ins that
speak the exact `Tool` contract the real clients do.

The scenarios are grouped by what they protect:

- `TestStatusTransitions` - the four orchestrator statuses and how a run
  reaches each.
- `TestStopSemantics`     - that each stop reason means what it says.
- `TestToolFailureModes`  - outages, empty results, malformed responses.
- `TestRetrySemantics`    - that a retry actually differs from its first try.
- `TestDelegation`        - NEEDS_AGENT, deterministically.
- `TestContinuation`      - slicing, checkpointing, resume, replay.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from agent.state.state import (
    final_gap_outcomes,
    open_gap_ids,
    unreconstructed_gap_ids,
)
from application.reconstruction_service import (
    ReconstructionOutcome,
    ReconstructionService,
)
from configuration.settings import (
    AzureSettings,
    BudgetSettings,
    ContinuationSettings,
    DatabaseSettings,
    EMBLEBISettings,
    LLMSettings,
    NCBISettings,
    NvidiaSettings,
    Settings,
)
from contracts.input import ReconstructionRequest
from contracts.observation import ObservationStatus
from contracts.output import GapReconstruction, ReconstructionStatus
from domain.exceptions import ExternalServiceError, InvalidSequenceError
from domain.models import AlignedPair, Alignment, Reference, Sequence
from observability.events import CollectingEmitter
from tools.blast.schemas import BlastSearchInput, BlastSearchOutput
from tools.contracts import Tool
from tools.evo.tool import EvolutionaryContextTool
from tools.mafft.schemas import AlignmentInput, AlignmentOutput
from tools.registry import ToolRegistry

pytestmark = pytest.mark.asyncio


# ---------------------------------------------------------------------------
# Fixtures: sequences and scripted tools
# ---------------------------------------------------------------------------

GAP_LEN = 12
LEFT = "ACGTTGCA" * 15
RIGHT = "TTACGGCA" * 15
TRUTH = "GGGGCCCCTTTT"

LEFT2 = "TTGGCCAA" * 15
RIGHT2 = "CCAATTGG" * 15
GAP2_LEN = 8


def gapped_one() -> Sequence:
    return Sequence.parse(
        "qa_one_gap", LEFT + "N" * GAP_LEN + RIGHT, organism="Mammuthus primigenius"
    )


def gapped_two() -> Sequence:
    return Sequence.parse(
        "qa_two_gaps",
        LEFT + "N" * GAP_LEN + RIGHT + LEFT2 + "N" * GAP2_LEN + RIGHT2,
        organism="Mammuthus primigenius",
    )


def make_reference(accession: str, organism: str = "Loxodonta africana") -> Reference:
    """A reference shaped as `tools/blast/mapper.to_references` builds one.

    `relatedness` is deliberately absent: the real mappers never set it, and it
    is what `evolutionary_context` exists to supply. A fixture that
    pre-populated it would hide the whole delegation path.
    """
    return Reference(
        accession=accession,
        organism=organism,
        residues=LEFT + TRUTH + RIGHT,
        identity=0.95,
        coverage=0.9,
        source="blast",
    )


def _spanning(payload: AlignmentInput, fills: dict[str, str], gap_len: int) -> Alignment:
    target_row = (
        payload.target_sequence[: payload.left_flank_length]
        + "-" * gap_len
        + payload.target_sequence[payload.left_flank_length :]
    )
    return Alignment(
        gap_id=payload.gap_id,
        pairs=[
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
        ],
        gap_column_start=payload.left_flank_length,
        gap_column_end=payload.left_flank_length + gap_len,
    )


class ScriptedBlast(Tool[BlastSearchInput, BlastSearchOutput]):
    """BLAST stand-in. `mode` selects which real-world outcome to reproduce."""

    name = "blast_search"
    description = "Scripted BLAST."
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
            raise ExternalServiceError("blast", "simulated outage")
        if self.mode == "zero_hits":
            return BlastSearchOutput(succeeded=True, references=[], total_hits=0)
        if self.mode == "malformed":
            return object()  # type: ignore[return-value]
        refs = [make_reference(f"REF_{len(self.calls)}_{i}") for i in range(3)]
        return BlastSearchOutput(succeeded=True, references=refs, total_hits=len(refs))


class ScriptedMafft(Tool[AlignmentInput, AlignmentOutput]):
    """MAFFT stand-in. `weak` makes the references genuinely disagree."""

    name = "mafft_align"
    description = "Scripted MAFFT."
    estimated_seconds = 0.0

    def __init__(
        self,
        mode: str = "good",
        gap_len: int = GAP_LEN,
        truth: str = TRUTH,
        delay: float = 0.0,
    ) -> None:
        self.mode = mode
        self.gap_len = gap_len
        self.truth = truth
        self.delay = delay
        self.calls: list[AlignmentInput] = []

    async def run(self, payload: AlignmentInput) -> AlignmentOutput:
        self.calls.append(payload)
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.mode == "outage":
            raise ExternalServiceError("mafft", "simulated outage")
        if self.mode == "no_span":
            # Aligned the flanks, inserted nothing across the gap.
            return AlignmentOutput(
                succeeded=True,
                alignment=Alignment(gap_id=payload.gap_id, pairs=[], gap_column_start=None),
                aligned_count=0,
            )
        if self.mode == "weak":
            # Each reference votes differently, so the consensus margin
            # collapses. Giving them all the SAME filling would be unanimous
            # agreement, which scores high and gets accepted.
            alphabet = ["A", "C", "G", "T"]
            fills = {
                accession: (alphabet[index % 4] * self.gap_len)[: self.gap_len]
                for index, accession in enumerate(payload.references)
            }
        else:
            fills = {accession: self.truth for accession in payload.references}

        alignment = _spanning(payload, fills, self.gap_len)
        return AlignmentOutput(
            succeeded=True, alignment=alignment, aligned_count=len(alignment.pairs)
        )


def settings_for(**overrides: Any) -> Settings:
    """Fully isolated settings - no real .env, no network, no LLM."""
    kwargs: dict[str, Any] = dict(
        _env_file=None,
        max_iterations=overrides.pop("max_iterations", 6),
        min_confidence=overrides.pop("min_confidence", 0.65),
        ncbi=NCBISettings(_env_file=None, contact_email="qa@umbrella.local"),
        embl_ebi=EMBLEBISettings(_env_file=None, contact_email="qa@umbrella.local"),
        azure=AzureSettings(_env_file=None),
        nvidia=NvidiaSettings(_env_file=None),
        llm=LLMSettings(_env_file=None),
        budgets=overrides.pop("budgets", BudgetSettings(_env_file=None)),
        continuation=overrides.pop(
            "continuation", ContinuationSettings(_env_file=None, yield_after_seconds=90.0)
        ),
        database=DatabaseSettings(_env_file=None),
    )
    kwargs.update(overrides)
    return Settings(**kwargs)  # type: ignore[arg-type]


async def run_agent(
    tools: ToolRegistry,
    sequence: Sequence | None = None,
    *,
    settings: Settings | None = None,
    trace_id: str | None = None,
    checkpointer: Any = None,
) -> tuple[ReconstructionOutcome, CollectingEmitter]:
    sequence = sequence or gapped_one()
    events = CollectingEmitter()
    service = ReconstructionService(
        settings or settings_for(), tools, events=events, checkpointer=checkpointer
    )
    request = ReconstructionRequest.from_agent_request(
        "Reconstruct the unresolved regions.",
        {
            "sequence": {"identifier": sequence.identifier, "residues": sequence.residues},
            "organism": sequence.organism,
        },
    )
    return await service.reconstruct(request, trace_id=trace_id), events


# ---------------------------------------------------------------------------


class TestStatusTransitions:
    """The four statuses the orchestrator's router branches on."""

    async def test_completed_accept(self) -> None:
        tools = ToolRegistry([ScriptedBlast("hits"), ScriptedMafft("good")])
        outcome, _ = await run_agent(tools)

        assert outcome.finished is True
        assert outcome.needs_agent is None
        assert outcome.result.stop_reason == "all_gaps_resolved"
        assert outcome.result.gaps[0].status is ReconstructionStatus.RECONSTRUCTED
        assert TRUTH in (outcome.result.reconstructed_sequence or "")

    async def test_completed_abstain_is_not_a_failure(self) -> None:
        """"These gaps cannot be reconstructed" is an answer, not an error."""
        tools = ToolRegistry([ScriptedBlast("hits"), ScriptedMafft("weak")])
        outcome, _ = await run_agent(tools, settings=settings_for(max_iterations=1))

        assert outcome.finished is True
        assert outcome.result.gaps[0].status is not ReconstructionStatus.RECONSTRUCTED

    async def test_partial_completion_reports_both_outcomes(self) -> None:
        good = ScriptedMafft("good")

        class MixedMafft(Tool[AlignmentInput, AlignmentOutput]):
            name = "mafft_align"
            description = "x"
            estimated_seconds = 0.0

            async def run(self, payload: AlignmentInput) -> AlignmentOutput:
                if payload.gap_id == "gap_2":
                    return await ScriptedMafft("no_span", gap_len=GAP2_LEN).run(payload)
                return await good.run(payload)

        tools = ToolRegistry([ScriptedBlast("hits"), MixedMafft()])
        outcome, _ = await run_agent(
            tools, gapped_two(), settings=settings_for(max_iterations=3)
        )

        statuses = {gap.gap_id: gap.status.value for gap in outcome.result.gaps}
        assert len(statuses) == 2
        assert any(status == "reconstructed" for status in statuses.values())
        assert any(status != "reconstructed" for status in statuses.values())

    async def test_failed_when_no_sequence_is_given(self) -> None:
        tools = ToolRegistry([ScriptedBlast("hits"), ScriptedMafft("good")])
        service = ReconstructionService(settings_for(), tools, events=CollectingEmitter())

        with pytest.raises(InvalidSequenceError):
            await service.reconstruct(
                ReconstructionRequest.from_agent_request("Reconstruct.", {})
            )

    async def test_failed_on_malformed_fasta(self) -> None:
        tools = ToolRegistry([ScriptedBlast("hits"), ScriptedMafft("good")])
        service = ReconstructionService(settings_for(), tools, events=CollectingEmitter())

        with pytest.raises(InvalidSequenceError):
            await service.reconstruct(
                ReconstructionRequest.from_agent_request(
                    "Reconstruct.",
                    {"sequence": {"identifier": "bad", "residues": "MKVLAAGIVGL"}},
                )
            )


class TestStopSemantics:
    """Each stop reason must mean what it says."""

    async def test_no_gaps_is_nothing_to_do_not_all_resolved(self) -> None:
        """A gapless sequence resolved nothing; calling that ALL_RESOLVED was a lie."""
        tools = ToolRegistry([ScriptedBlast("hits"), ScriptedMafft("good")])
        ungapped = Sequence.parse("qa_none", "ACGT" * 40, organism="Testus organismus")

        outcome, _ = await run_agent(tools, ungapped)

        assert outcome.finished is True
        assert outcome.result.stop_reason == "nothing_to_do"
        assert outcome.result.gaps == []

    async def test_all_resolved_requires_actually_reconstructed(self) -> None:
        tools = ToolRegistry([ScriptedBlast("hits"), ScriptedMafft("good")])
        outcome, _ = await run_agent(tools)

        assert outcome.result.stop_reason == "all_gaps_resolved"
        assert all(
            gap.status is ReconstructionStatus.RECONSTRUCTED for gap in outcome.result.gaps
        )

    async def test_budget_exhausted_reports_partial_results(self) -> None:
        tools = ToolRegistry([ScriptedBlast("hits"), ScriptedMafft("good")])
        outcome, _ = await run_agent(
            tools,
            settings=settings_for(budgets=BudgetSettings(_env_file=None, max_tool_calls=1)),
        )

        assert outcome.result.stop_reason == "budget_exhausted"
        assert outcome.result.budget["tool_calls"] <= 1
        assert outcome.result.gaps[0].status is not ReconstructionStatus.RECONSTRUCTED

    async def test_a_stalled_run_stops_before_max_iterations(self) -> None:
        """Progress is measured per round. Reading accumulated state instead
        would keep the run alive until MAX_ITERATIONS once any round had
        succeeded - which is every run that ever found one reference."""
        tools = ToolRegistry([ScriptedBlast("hits"), ScriptedMafft("no_span")])
        outcome, _ = await run_agent(tools, settings=settings_for(max_iterations=8))

        assert outcome.result.iterations < 8, "the loop burned every iteration on a stall"
        assert outcome.result.stop_reason != "max_iterations_reached"


# These are pure functions, so they need no event loop - the module-level
# asyncio mark would otherwise warn on every one of them.
@pytest.mark.filterwarnings("ignore::pytest.PytestWarning")
class TestStateHelpers:
    """The predicates the loop's routing depends on."""

    def test_an_unresolved_entry_is_not_a_final_outcome(self) -> None:
        state = {
            "gap_contexts": [],
            "skipped": {},
            "reconstructions": {
                "gap_1": GapReconstruction(
                    gap_id="gap_1",
                    start=0,
                    end=4,
                    length=4,
                    status=ReconstructionStatus.UNRESOLVED,
                )
            },
            "verdicts": {},
        }

        assert final_gap_outcomes(state) == {}

    def test_an_abstain_verdict_settles_a_gap(self) -> None:
        state = {"skipped": {}, "reconstructions": {}, "verdicts": {"gap_1": "abstain"}}

        assert final_gap_outcomes(state)["gap_1"] == "abstained"

    def test_escalation_covers_abstained_gaps_but_not_skipped_ones(self) -> None:
        """A gap the critic gave up on is exactly what to ask Evolution about;
        one skipped for its own shape is not."""

        class Ctx:
            def __init__(self, identifier: str) -> None:
                self.identifier = identifier

        state = {
            "gap_contexts": [Ctx("gap_1"), Ctx("gap_2")],
            "skipped": {"gap_2": "too long"},
            "reconstructions": {},
            "verdicts": {"gap_1": "abstain"},
        }

        assert unreconstructed_gap_ids(state) == {"gap_1"}
        assert open_gap_ids(state) == set()


class TestToolFailureModes:
    async def test_zero_blast_hits_is_empty_not_failed(self) -> None:
        tools = ToolRegistry([ScriptedBlast("zero_hits"), ScriptedMafft("good")])
        outcome, _ = await run_agent(tools, settings=settings_for(max_iterations=3))

        blast = [o for o in outcome.result.observations if o.tool == "blast_search"]
        assert blast
        assert all(o.status is ObservationStatus.EMPTY for o in blast)

    async def test_external_outage_leaves_the_run_completable(self) -> None:
        tools = ToolRegistry([ScriptedBlast("outage"), ScriptedMafft("good")])
        outcome, _ = await run_agent(tools, settings=settings_for(max_iterations=2))

        failed = [o for o in outcome.result.observations if o.status is ObservationStatus.FAILED]
        assert outcome.finished is True
        assert failed
        assert outcome.result.gaps[0].status is not ReconstructionStatus.RECONSTRUCTED

    async def test_malformed_tool_output_is_a_recorded_contract_violation(self) -> None:
        """A tool returning the wrong type must not look like an empty result."""
        tools = ToolRegistry([ScriptedBlast("malformed"), ScriptedMafft("good")])
        outcome, _ = await run_agent(tools, settings=settings_for(max_iterations=2))

        blast = [o for o in outcome.result.observations if o.tool == "blast_search"]
        assert outcome.finished is True, "a broken tool must not crash the run"
        assert all(o.status is ObservationStatus.FAILED for o in blast)
        assert any(
            o.diagnostics.get("contract_violation") == "tool_output_type" for o in blast
        )

    async def test_weak_alignment_is_not_silently_accepted(self) -> None:
        tools = ToolRegistry([ScriptedBlast("hits"), ScriptedMafft("weak")])
        outcome, _ = await run_agent(tools, settings=settings_for(max_iterations=1))

        assert outcome.result.gaps[0].status is not ReconstructionStatus.RECONSTRUCTED

    async def test_non_spanning_alignment_yields_unresolved(self) -> None:
        tools = ToolRegistry([ScriptedBlast("hits"), ScriptedMafft("no_span")])
        outcome, _ = await run_agent(tools, settings=settings_for(max_iterations=1))

        assert outcome.result.gaps[0].status is ReconstructionStatus.UNRESOLVED


class TestRetrySemantics:
    async def test_a_retry_widens_the_search_rather_than_repeating_it(self) -> None:
        """A byte-identical retry could only ever return the same answer."""
        blast = ScriptedBlast("zero_hits")
        tools = ToolRegistry([blast, ScriptedMafft("good")])
        outcome, _ = await run_agent(tools, settings=settings_for(max_iterations=4))

        assert len(blast.calls) >= 2, "no retry was attempted"
        first, second = blast.calls[0], blast.calls[1]
        assert (first.expect, first.max_hits, first.database) != (
            second.expect,
            second.max_hits,
            second.database,
        )
        assert second.expect > first.expect, "the retry did not loosen the e-value"
        assert any(
            o.relaxed for o in outcome.result.observations if o.tool == "blast_search"
        ), "the retry was not recorded as relaxed in the audit trail"

    async def test_a_non_spanning_alignment_does_not_satisfy_the_precondition(self) -> None:
        """If it did, the retry would never be planned - which is what happened."""
        mafft = ScriptedMafft("no_span")
        tools = ToolRegistry([ScriptedBlast("hits"), mafft])
        await run_agent(tools, settings=settings_for(max_iterations=6))

        assert len(mafft.calls) == 2, (
            f"expected one attempt plus one retry, got {len(mafft.calls)}"
        )

    async def test_retries_are_bounded_at_two_attempts(self) -> None:
        mafft = ScriptedMafft("no_span")
        tools = ToolRegistry([ScriptedBlast("hits"), mafft])
        await run_agent(tools, settings=settings_for(max_iterations=8))

        assert len(mafft.calls) <= 2


class TestDelegation:
    """NEEDS_AGENT must work without an LLM, or it does not work when it matters."""

    async def test_escalates_to_evolution_without_any_llm(self) -> None:
        tools = ToolRegistry(
            [ScriptedBlast("hits"), ScriptedMafft("weak"), EvolutionaryContextTool()]
        )
        outcome, events = await run_agent(tools, settings=settings_for(max_iterations=4))

        evo_calls = [e for e in events.events if e.data.get("tool") == "evolutionary_context"]
        assert evo_calls, "the deterministic planner never invoked evolutionary_context"
        assert outcome.needs_agent == "Evolution"
        assert outcome.result.stop_reason == "delegated_to_another_agent"
        assert outcome.prompt_to_target_agent
        assert "Mammuthus" in outcome.prompt_to_target_agent

    async def test_escalation_names_a_gap_that_never_produced_a_candidate(self) -> None:
        tools = ToolRegistry(
            [ScriptedBlast("hits"), ScriptedMafft("no_span"), EvolutionaryContextTool()]
        )
        outcome, _ = await run_agent(tools, settings=settings_for(max_iterations=4))

        assert outcome.needs_agent == "Evolution"
        assert outcome.prompt_to_target_agent
        assert "gap_1" in outcome.prompt_to_target_agent

    async def test_escalation_is_not_gated_by_the_slice_budget(self) -> None:
        """NEEDS_AGENT is routed by the capability resolver and does not consume
        a CONTINUE retry, so the last slice must still be able to ask for help."""
        tools = ToolRegistry(
            [ScriptedBlast("hits"), ScriptedMafft("weak"), EvolutionaryContextTool()]
        )
        outcome, _ = await run_agent(
            tools,
            settings=settings_for(
                max_iterations=4,
                continuation=ContinuationSettings(_env_file=None, max_slices=1),
            ),
        )

        assert outcome.needs_agent == "Evolution"

    async def test_escalation_carries_findings_so_the_loop_guard_sees_progress(self) -> None:
        """The orchestrator force-fails an escalation that adds no context keys."""
        tools = ToolRegistry(
            [ScriptedBlast("hits"), ScriptedMafft("weak"), EvolutionaryContextTool()]
        )
        outcome, _ = await run_agent(tools, settings=settings_for(max_iterations=4))

        assert outcome.result.gaps, "no findings travelled with the escalation"

    async def test_no_escalation_when_relatedness_already_separates_references(self) -> None:
        """Same-genus references need no phylogenetic help."""

        class SameGenusBlast(Tool[BlastSearchInput, BlastSearchOutput]):
            name = "blast_search"
            description = "x"
            estimated_seconds = 0.0

            async def run(self, payload: BlastSearchInput) -> BlastSearchOutput:
                return BlastSearchOutput(
                    succeeded=True,
                    references=[
                        make_reference("REF_A", organism="Mammuthus columbi"),
                        make_reference("REF_B", organism="Mammuthus trogontherii"),
                    ],
                    total_hits=2,
                )

        tools = ToolRegistry([SameGenusBlast(), ScriptedMafft("good"), EvolutionaryContextTool()])
        outcome, _ = await run_agent(tools, settings=settings_for(max_iterations=4))

        assert outcome.needs_agent is None


class TestContinuation:
    async def test_yield_produces_a_resumable_continuation(self) -> None:
        from langgraph.checkpoint.memory import InMemorySaver

        tools = ToolRegistry([ScriptedBlast("hits", delay=0.05), ScriptedMafft("good")])
        settings = settings_for(
            continuation=ContinuationSettings(_env_file=None, yield_after_seconds=0.001)
        )
        outcome, _ = await run_agent(
            tools, settings=settings, trace_id="qa-yield", checkpointer=InMemorySaver()
        )

        assert outcome.finished is False
        assert outcome.continuation_reason

    async def test_resume_does_not_repeat_completed_tool_calls(self) -> None:
        from langgraph.checkpoint.memory import InMemorySaver

        # BLAST answers instantly, so its result is genuinely paid for. MAFFT
        # is what exhausts the slice - and being aborted, it is *not* paid for,
        # which is why only BLAST is expected to be spared on resume.
        checkpointer = InMemorySaver()
        blast = ScriptedBlast("hits", delay=0.0)
        tools = ToolRegistry([blast, ScriptedMafft("good", delay=30.0)])
        settings = settings_for(
            continuation=ContinuationSettings(_env_file=None, yield_after_seconds=0.5)
        )

        first, _ = await run_agent(
            tools, settings=settings, trace_id="qa-resume", checkpointer=checkpointer
        )
        assert first.finished is False
        calls_after_first = len(blast.calls)

        await run_agent(
            tools, settings=settings, trace_id="qa-resume", checkpointer=checkpointer
        )

        assert len(blast.calls) == calls_after_first, (
            "the resumed slice redid work the first slice had already paid for"
        )

    async def test_the_last_slice_finishes_rather_than_asking_for_another(self) -> None:
        from langgraph.checkpoint.memory import InMemorySaver

        tools = ToolRegistry([ScriptedBlast("hits", delay=0.05), ScriptedMafft("good")])
        settings = settings_for(
            continuation=ContinuationSettings(
                _env_file=None, yield_after_seconds=0.001, max_slices=1
            )
        )
        outcome, _ = await run_agent(
            tools, settings=settings, trace_id="qa-last", checkpointer=InMemorySaver()
        )

        assert outcome.finished is True
        assert outcome.continuation_reason is None

    async def test_a_replayed_trace_id_does_no_further_work(self) -> None:
        """A completed run re-called with the same trace id must short-circuit,
        not enter another plan/critique round to rediscover it was done."""
        from langgraph.checkpoint.memory import InMemorySaver

        checkpointer = InMemorySaver()
        blast = ScriptedBlast("hits")
        tools = ToolRegistry([blast, ScriptedMafft("good")])

        first, _ = await run_agent(tools, trace_id="qa-replay", checkpointer=checkpointer)
        assert first.finished is True
        assert first.result.stop_reason == "all_gaps_resolved"
        calls_after_first = len(blast.calls)

        second, events = await run_agent(
            tools, trace_id="qa-replay", checkpointer=checkpointer
        )

        assert len(blast.calls) == calls_after_first, "the replay re-ran a real tool call"
        assert not [e for e in events.events if e.type.value == "plan_created"], (
            "the replay entered a planning round for a run that was already finished"
        )
