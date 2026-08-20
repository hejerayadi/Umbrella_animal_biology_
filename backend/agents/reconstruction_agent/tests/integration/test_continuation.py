"""Slicing a run across HTTP calls, and resuming it.

The claim under test: a reconstruction too slow for one 120 s call yields a
CONTINUE, and the next call picks up where it stopped instead of redoing the
work. Without that, every multi-gap run fails after three retries.

Runs the real graph against a stub tool registry, so it exercises the actual
LangGraph wiring and checkpointer rather than a mock of them.
"""
from __future__ import annotations

import asyncio
import time

import pytest

from agent.state.state import initial_state, is_last_slice
from application.run_agent import AgentRunner
from configuration.settings import (
    BudgetSettings,
    ContinuationSettings,
    EMBLEBISettings,
    LLMSettings,
    NCBISettings,
    Settings,
)
from domain.models import Reference, Sequence
from tools.contracts import Tool, ToolInput, ToolOutput

pytestmark = pytest.mark.asyncio

# One gap with generous flanks on both sides.
GAPPED = "ACGTTGCA" * 30 + "N" * 12 + "TTACGGCA" * 30


class _SearchInput(ToolInput):
    sequence: str = ""
    gap_id: str | None = None
    database: str = ""
    program: str = ""
    max_hits: int = 0
    expect: float = 0.0


class _SearchOutput(ToolOutput):
    references: list = []


class SlowBlast(Tool[_SearchInput, _SearchOutput]):
    """A BLAST stand-in that takes long enough to exhaust the slice budget."""

    name = "blast_search"
    description = "Stub BLAST."
    estimated_seconds = 1.0

    def __init__(self, delay: float) -> None:
        self._delay = delay
        self.calls = 0

    async def run(self, payload: _SearchInput) -> _SearchOutput:
        self.calls += 1
        await asyncio.sleep(self._delay)
        return _SearchOutput(
            succeeded=True,
            references=[Reference(accession=f"REF_{self.calls}", residues="ACGT" * 20)],
        )


class _AlignInput(ToolInput):
    gap_id: str | None = None
    sequence: str = ""
    references: list = []


class _AlignOutput(ToolOutput):
    alignment: object | None = None
    aligned_count: int = 0


class StalledMafft(Tool[_AlignInput, _AlignOutput]):
    """An alignment that outlives any slice, so the deadline always aborts it.

    Distinct from a *slow* tool that still finishes: this is the case the
    orchestrator used to see as an unreachable agent, because the yield check
    only runs between nodes and could not interrupt the call.
    """

    name = "mafft_align"
    description = "Stub MAFFT that never returns in time."
    estimated_seconds = 1.0

    def __init__(self, delay: float = 30.0) -> None:
        self._delay = delay
        self.calls = 0

    async def run(self, payload: _AlignInput) -> _AlignOutput:
        self.calls += 1
        await asyncio.sleep(self._delay)
        return _AlignOutput(succeeded=True)


def build_settings(**continuation: object) -> Settings:
    """Settings with no external credentials and no LLM.

    `_env_file=None` keeps a developer's real `.env` from deciding whether
    these pass.
    """
    return Settings(
        _env_file=None,  # type: ignore[call-arg]
        max_iterations=4,
        ncbi=NCBISettings(_env_file=None),  # type: ignore[call-arg]
        embl_ebi=EMBLEBISettings(_env_file=None),  # type: ignore[call-arg]
        llm=LLMSettings(_env_file=None),  # type: ignore[call-arg]
        budgets=BudgetSettings(_env_file=None),  # type: ignore[call-arg]
        continuation=ContinuationSettings(_env_file=None, **continuation),  # type: ignore[call-arg]
    )


def make_runner(settings: Settings, registry: object, checkpointer: object) -> AgentRunner:
    from observability.events import CollectingEmitter

    return AgentRunner(
        settings, registry, CollectingEmitter(), checkpointer=checkpointer  # type: ignore[arg-type]
    )


@pytest.fixture
def checkpointer() -> object:
    from langgraph.checkpoint.memory import InMemorySaver

    return InMemorySaver()


@pytest.fixture
def target() -> Sequence:
    return Sequence.parse("slice_seq", GAPPED, organism="Testus organismus")


class TestSlicing:
    async def test_a_slow_slice_yields_instead_of_finishing(
        self, checkpointer: object, target: Sequence
    ) -> None:
        from tools.registry import ToolRegistry

        # Yield almost immediately, so the first tool round exhausts the slice.
        settings = build_settings(yield_after_seconds=0.01)
        registry = ToolRegistry([SlowBlast(delay=0.05)])
        runner = make_runner(settings, registry, checkpointer)

        outcome = await runner.run_slice(
            run_id="run-1",
            trace_id="trace-abc",
            instruction="Reconstruct.",
            target=target,
            organism="Testus organismus",
            requested_organisms=[],
        )

        assert not outcome.finished
        assert outcome.continuation_reason is not None
        assert "gap" in outcome.continuation_reason.lower()

    async def test_a_tool_that_outlives_the_slice_is_aborted_not_waited_on(
        self, checkpointer: object, target: Sequence
    ) -> None:
        """The regression that made the orchestrator call the agent unreachable.

        MAFFT at EMBL-EBI polls for up to 600 s. The yield check runs between
        graph nodes, so before the call itself was bounded, one such job held
        the slice open past the orchestrator's 120 s read timeout - and every
        finding in that slice was discarded as a transport failure.
        """
        from tools.registry import ToolRegistry

        window = 0.4
        mafft = StalledMafft(delay=30.0)
        settings = build_settings(yield_after_seconds=window)
        runner = make_runner(settings, ToolRegistry([SlowBlast(delay=0.0), mafft]), checkpointer)

        started = time.monotonic()
        outcome = await runner.run_slice(
            run_id="run-1",
            trace_id="trace-deadline",
            instruction="Reconstruct.",
            target=target,
            organism="Testus organismus",
            requested_organisms=[],
        )
        elapsed = time.monotonic() - started

        assert mafft.calls, "the alignment was never attempted"
        # Bounded by the slice, not by the tool: nowhere near its 30 s delay.
        assert elapsed < 5.0, f"the slice ran for {elapsed:.1f}s"
        # And it asks for another slice rather than reporting a failure.
        assert not outcome.finished
        assert outcome.continuation_reason

    async def test_the_next_slice_resumes_rather_than_restarting(
        self, checkpointer: object, target: Sequence
    ) -> None:
        """The whole point: work already paid for must not be redone."""
        from tools.registry import ToolRegistry

        # BLAST answers instantly - that result is genuinely paid for, and is
        # what must survive. MAFFT is what exhausts the slice.
        settings = build_settings(yield_after_seconds=0.5)
        blast = SlowBlast(delay=0.0)
        registry = ToolRegistry([blast, StalledMafft()])
        runner = make_runner(settings, registry, checkpointer)

        first = await runner.run_slice(
            run_id="run-1",
            trace_id="trace-abc",
            instruction="Reconstruct.",
            target=target,
            organism="Testus organismus",
            requested_organisms=[],
        )
        assert not first.finished

        second = await runner.run_slice(
            run_id="run-2",
            trace_id="trace-abc",
            instruction="Reconstruct.",
            target=target,
            organism="Testus organismus",
            requested_organisms=[],
        )

        # Gaps were detected once, in the first slice, and carried over.
        assert second.state.get("gap_contexts")
        # The second slice is a later slice of the same run, not a fresh one.
        assert second.state.get("slice_index", 0) >= first.state.get("slice_index", 0)
        # References gathered in slice one survived.
        assert second.state.get("references")

    async def test_a_different_trace_id_starts_a_fresh_run(
        self, checkpointer: object, target: Sequence
    ) -> None:
        """Checkpoints are keyed by trace id; runs must not bleed into each other.

        Asserted on accumulated evidence rather than `slice_index`: after a
        yield that index is incremented to name the *next* slice, so a fresh
        run that also yields legitimately ends on the same number.
        """
        from tools.registry import ToolRegistry

        settings = build_settings(yield_after_seconds=0.01)
        runner = make_runner(settings, ToolRegistry([SlowBlast(delay=0.05)]), checkpointer)

        async def slice_for(trace_id: str, run_id: str) -> object:
            return await runner.run_slice(
                run_id=run_id,
                trace_id=trace_id,
                instruction="Reconstruct.",
                target=target,
                organism=None,
                requested_organisms=[],
            )

        first = await slice_for("trace-one", "run-1")
        resumed = await slice_for("trace-one", "run-2")
        fresh = await slice_for("trace-two", "run-3")

        one_slice = len(first.state.get("observations") or [])  # type: ignore[attr-defined]
        # Resuming the same trace keeps accumulating; a new trace starts over.
        assert len(resumed.state.get("observations") or []) > one_slice  # type: ignore[attr-defined]
        assert len(fresh.state.get("observations") or []) == one_slice  # type: ignore[attr-defined]

    async def test_the_last_slice_finishes_rather_than_asking_for_another(
        self, checkpointer: object, target: Sequence
    ) -> None:
        """A fifth CONTINUE is converted to FAILED, discarding every finding."""
        from tools.registry import ToolRegistry

        settings = build_settings(yield_after_seconds=0.01, max_slices=1)
        runner = make_runner(settings, ToolRegistry([SlowBlast(delay=0.05)]), checkpointer)

        outcome = await runner.run_slice(
            run_id="run-1",
            trace_id="trace-last",
            instruction="Reconstruct.",
            target=target,
            organism=None,
            requested_organisms=[],
        )

        assert outcome.finished
        assert outcome.continuation_reason is None

    async def test_a_fast_run_finishes_in_one_slice(
        self, checkpointer: object, target: Sequence
    ) -> None:
        from tools.registry import ToolRegistry

        settings = build_settings(yield_after_seconds=120.0)
        runner = make_runner(settings, ToolRegistry([SlowBlast(delay=0.0)]), checkpointer)

        outcome = await runner.run_slice(
            run_id="run-1",
            trace_id="trace-fast",
            instruction="Reconstruct.",
            target=target,
            organism=None,
            requested_organisms=[],
        )

        assert outcome.finished
        assert outcome.state.get("slice_index", 0) == 0


@pytest.mark.filterwarnings("ignore::pytest.PytestWarning")
class TestSliceArithmetic:
    async def test_the_final_granted_slice_is_recognised(self) -> None:
        state = initial_state(
            run_id="r",
            trace_id="t",
            instruction="",
            target=Sequence.parse("s", "ACGT"),
            organism=None,
            requested_organisms=[],
            max_iterations=6,
            max_slices=4,
        )

        assert not is_last_slice({**state, "slice_index": 0})
        assert not is_last_slice({**state, "slice_index": 2})
        # Index 3 is the fourth and final call the orchestrator makes.
        assert is_last_slice({**state, "slice_index": 3})
