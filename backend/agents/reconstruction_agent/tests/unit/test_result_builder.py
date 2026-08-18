"""Assembling final state into the result the orchestrator receives."""
from __future__ import annotations

from application.result_builder import ResultBuilder
from contracts.output import GapReconstruction, ReconstructionStatus
from domain.models import Gap, GapContext, Sequence


def make_state(**overrides: object) -> dict:
    target = Sequence.parse("seq", "ACGT" * 25 + "N" * 8 + "TTAC" * 25)
    context = GapContext(
        gap=Gap("gap_1", 100, 108),
        left_flank="A" * 100,
        right_flank="C" * 100,
    )
    state = {
        "target": target,
        "organism": "Testus organismus",
        "gap_contexts": [context],
        "skipped": {},
        "reconstructions": {},
        "tool_calls": [],
        "warnings": [],
        "errors": [],
        "iteration": 1,
    }
    state.update(overrides)
    return state


class TestResultBuilder:
    def test_attempted_but_unresolved_gaps_still_appear(self) -> None:
        """Regression: an attempted gap that produced nothing was omitted
        entirely, and the summary then reported a gapped sequence as having no
        unresolved regions at all."""
        result = ResultBuilder().build(make_state())

        assert len(result.gaps) == 1
        assert result.gaps[0].status is ReconstructionStatus.UNRESOLVED
        assert "no unresolved regions" not in result.summary

    def test_skipped_gaps_are_reported_with_their_reason(self) -> None:
        result = ResultBuilder().build(
            make_state(skipped={"gap_1": "Gap is 8000 bases, above the limit."})
        )

        assert len(result.gaps) == 1
        assert result.gaps[0].status is ReconstructionStatus.SKIPPED
        assert "above the limit" in (result.gaps[0].explanation or "")

    def test_a_gap_is_never_reported_twice(self) -> None:
        reconstruction = GapReconstruction(
            gap_id="gap_1",
            start=100,
            end=108,
            length=8,
            status=ReconstructionStatus.RECONSTRUCTED,
            reconstructed_sequence="GGGGGGGG",
            confidence=0.8,
        )

        result = ResultBuilder().build(
            make_state(reconstructions={"gap_1": reconstruction})
        )

        assert len(result.gaps) == 1
        assert result.gaps[0].status is ReconstructionStatus.RECONSTRUCTED

    def test_no_rebuilt_sequence_when_nothing_changed(self) -> None:
        """Echoing the input back as a 'reconstruction' would misrepresent it."""
        result = ResultBuilder().build(make_state())

        assert result.reconstructed_sequence is None

    def test_rebuilt_sequence_is_returned_when_a_gap_was_filled(self) -> None:
        reconstruction = GapReconstruction(
            gap_id="gap_1",
            start=100,
            end=108,
            length=8,
            status=ReconstructionStatus.RECONSTRUCTED,
            reconstructed_sequence="GGGGGGGG",
            confidence=0.8,
        )

        result = ResultBuilder().build(
            make_state(reconstructions={"gap_1": reconstruction})
        )

        assert result.reconstructed_sequence is not None
        assert "N" not in result.reconstructed_sequence

    def test_overall_confidence_is_the_weakest_filled_gap(self) -> None:
        contexts = [
            GapContext(gap=Gap("gap_1", 100, 104), left_flank="A" * 100, right_flank="C" * 100),
            GapContext(gap=Gap("gap_2", 200, 204), left_flank="A" * 100, right_flank="C" * 100),
        ]
        reconstructions = {
            f"gap_{index}": GapReconstruction(
                gap_id=f"gap_{index}",
                start=start,
                end=start + 4,
                length=4,
                status=ReconstructionStatus.RECONSTRUCTED,
                reconstructed_sequence="ACGT",
                confidence=confidence,
            )
            for index, (start, confidence) in enumerate([(100, 0.9), (200, 0.6)], start=1)
        }

        result = ResultBuilder().build(
            make_state(gap_contexts=contexts, reconstructions=reconstructions)
        )

        assert result.overall_confidence == 0.6

    def test_early_stop_is_surfaced_as_a_warning(self) -> None:
        result = ResultBuilder().build(make_state(stop_reason="max_iterations_reached"))

        assert any("stopped early" in warning for warning in result.warnings)

    def test_output_payload_is_json_serialisable(self) -> None:
        payload = ResultBuilder.to_output_payload(ResultBuilder().build(make_state()))

        assert isinstance(payload["reconstruction"]["completed_at"], str)
        assert payload["reconstruction"]["gaps"][0]["status"] == "unresolved"

    def test_output_payload_is_namespaced(self) -> None:
        """The orchestrator merges this flat into the context every agent
        reads, so bare keys like `summary` would collide with another agent's."""
        payload = ResultBuilder.to_output_payload(ResultBuilder().build(make_state()))

        assert set(payload) == {
            "reconstruction",
            "reconstruction_summary",
            "reconstruction_sequence",
        }

    def test_budget_and_stop_reason_are_reported(self) -> None:
        """A partial answer has to explain itself rather than look truncated."""
        result = ResultBuilder().build(
            make_state(
                stop_reason="budget_exhausted",
                budget_tool_calls=12,
                budget_llm_tokens=4200,
                slice_index=2,
            )
        )

        assert result.stop_reason == "budget_exhausted"
        assert result.budget["tool_calls"] == 12
        assert result.budget["llm_tokens"] == 4200
        assert result.slices == 3
