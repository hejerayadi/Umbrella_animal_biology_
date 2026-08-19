"""Turns finished graph state into the result the orchestrator receives.

The only place that knows both the agent's internal state and the shape the
orchestrator parses. Everything upstream works in domain terms; everything
downstream sees `AgentResult`.
"""
from __future__ import annotations

from typing import Any

from agent.reasoning.evidence_synthesizer import EvidenceSynthesizer
from agent.state.state import ReconstructionState
from configuration.logging import get_logger
from contracts.output import GapReconstruction, ReconstructionResult, ReconstructionStatus

_log = get_logger(__name__)


class ResultBuilder:
    """Assembles a `ReconstructionResult` from final state."""

    def __init__(self, synthesizer: EvidenceSynthesizer | None = None) -> None:
        self._synthesizer = synthesizer or EvidenceSynthesizer()

    def build(self, state: ReconstructionState) -> ReconstructionResult:
        target = state["target"]
        skipped = state.get("skipped") or {}

        reconstructions = list((state.get("reconstructions") or {}).values())
        reconstructions.extend(self._skipped_entries(state, skipped))
        reconstructions.extend(self._unresolved_entries(state, skipped))
        reconstructions.sort(key=lambda item: item.start)

        applied = self._synthesizer.apply(target, reconstructions)

        confidences = [
            item.confidence
            for item in reconstructions
            if item.status is ReconstructionStatus.RECONSTRUCTED
        ]

        return ReconstructionResult(
            sequence_id=target.identifier,
            organism=state.get("organism"),
            original_length=len(target),
            # Only report a rebuilt sequence when something actually changed;
            # echoing the input back as a "reconstruction" would misrepresent
            # a run that resolved nothing.
            reconstructed_sequence=(
                applied.residues if applied.residues != target.residues else None
            ),
            gaps=reconstructions,
            overall_confidence=(
                self._aggregate(confidences) if confidences else 0.0
            ),
            summary=self._synthesizer.summarise(
                target,
                reconstructions,
                skipped=skipped,
                iterations=state.get("iteration", 0),
            ),
            iterations=state.get("iteration", 0),
            tools_used=sorted(
                {call["tool"] for call in (state.get("tool_calls") or []) if call.get("tool")}
            ),
            warnings=self._warnings(state),
            stop_reason=state.get("stop_reason"),
            slices=state.get("slice_index", 0) + 1,
            budget={
                "tool_calls": state.get("budget_tool_calls", 0),
                "tool_calls_by_name": dict(state.get("budget_tool_calls_by_name") or {}),
                "llm_tokens": state.get("budget_llm_tokens", 0),
            },
            observations=list(state.get("observations") or []),
        )

    @staticmethod
    def _aggregate(confidences: list[float]) -> float:
        """Overall confidence is the weakest reconstruction in the set."""
        return round(min(confidences), 4)

    @staticmethod
    def _skipped_entries(
        state: ReconstructionState, skipped: dict[str, str]
    ) -> list[GapReconstruction]:
        """Report declined gaps explicitly rather than omitting them."""
        contexts = {context.identifier: context for context in state.get("gap_contexts") or []}
        entries: list[GapReconstruction] = []

        for gap_id, reason in skipped.items():
            context = contexts.get(gap_id)
            if context is None:
                continue
            entries.append(
                GapReconstruction(
                    gap_id=gap_id,
                    start=context.gap.start,
                    end=context.gap.end,
                    length=context.gap.length,
                    status=ReconstructionStatus.SKIPPED,
                    explanation=reason,
                )
            )
        return entries

    @staticmethod
    def _unresolved_entries(
        state: ReconstructionState, skipped: dict[str, str]
    ) -> list[GapReconstruction]:
        """Report attempted gaps that produced nothing.

        A gap the agent tried and failed on must appear in the output. Without
        this it would be omitted entirely, and the summary - which counts what
        it is given - would then report a sequence full of holes as having no
        unresolved regions at all.
        """
        resolved = state.get("reconstructions") or {}
        entries: list[GapReconstruction] = []

        for context in state.get("gap_contexts") or []:
            gap_id = context.identifier
            if gap_id in resolved or gap_id in skipped:
                continue
            entries.append(
                GapReconstruction(
                    gap_id=gap_id,
                    start=context.gap.start,
                    end=context.gap.end,
                    length=context.gap.length,
                    status=ReconstructionStatus.UNRESOLVED,
                    explanation=(
                        "The agent attempted this region but gathered no reference "
                        "evidence spanning it."
                    ),
                )
            )
        return entries

    @staticmethod
    def _warnings(state: ReconstructionState) -> list[str]:
        """Everything the caller should know that is not an outright failure."""
        warnings = list(state.get("warnings") or [])
        warnings.extend(state.get("errors") or [])

        stop_reason = state.get("stop_reason")
        if stop_reason in ("max_iterations_reached", "no_progress"):
            warnings.append(
                f"The agent stopped early ({stop_reason}); some gaps may be resolvable "
                "with a higher iteration budget."
            )
        return warnings

    @staticmethod
    def to_output_payload(result: ReconstructionResult) -> dict[str, Any]:
        """The dict placed in `AgentResult.output`.

        Namespaced under `reconstruction` because the orchestrator merges this
        straight into the context every agent shares
        (`worker_node.py` -> `updates["context"]`). Flat keys like `summary`
        or `organism` would collide with whatever another agent wrote.

        Two flat aliases survive at the top level: the Responder renders
        context keys into its prompt, and these are the two a human answer
        actually needs. They are prefixed so they still cannot clash.

        `mode="json"` so datetimes and enums are already serialisable - the
        orchestrator passes this straight into a JSON response.
        """
        payload = result.model_dump(mode="json")
        return {
            "reconstruction": payload,
            "reconstruction_summary": result.summary,
            "reconstruction_sequence": result.reconstructed_sequence,
        }
