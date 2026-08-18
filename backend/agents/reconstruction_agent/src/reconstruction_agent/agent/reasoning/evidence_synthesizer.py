"""Assembles the run's evidence into the summary a reader actually gets.

The per-gap explanations say why each filling was chosen; this says what the
run as a whole established, including what it could not do.
"""
from __future__ import annotations

from dataclasses import dataclass

from ...contracts.output import GapReconstruction, ReconstructionStatus
from ...domain.models import Sequence
from ...domain.models.sequence import UNKNOWN_BASE


@dataclass(frozen=True, slots=True)
class EvidenceSynthesizer:
    """Builds the final sequence and its narrative summary."""

    def apply(
        self, target: Sequence, reconstructions: list[GapReconstruction]
    ) -> Sequence:
        """The target with every confident reconstruction spliced in.

        Applied right-to-left so that earlier gaps' offsets stay valid as later
        ones change the string's length - a reconstruction need not be the same
        length as the gap it fills.

        Gaps that were not resolved keep their original N runs, so the output
        stays directly comparable to the input.
        """
        residues = target.residues

        for reconstruction in sorted(reconstructions, key=lambda item: item.start, reverse=True):
            if reconstruction.status is not ReconstructionStatus.RECONSTRUCTED:
                continue
            if not reconstruction.reconstructed_sequence:
                continue
            residues = (
                residues[: reconstruction.start]
                + reconstruction.reconstructed_sequence
                + residues[reconstruction.end :]
            )

        return target.with_residues(residues)

    def summarise(
        self,
        target: Sequence,
        reconstructions: list[GapReconstruction],
        *,
        skipped: dict[str, str],
        iterations: int,
    ) -> str:
        """A plain-language account of the run.

        Deliberately leads with what was not resolved when that dominates: a
        summary that opens with three successes and buries nine failures
        misrepresents the run.
        """
        by_status = {status: 0 for status in ReconstructionStatus}
        for reconstruction in reconstructions:
            by_status[reconstruction.status] += 1

        attempted = len(reconstructions)
        resolved = by_status[ReconstructionStatus.RECONSTRUCTED]
        total_gaps = attempted + len(skipped)

        if total_gaps == 0:
            return (
                f"{target.identifier} contains no unresolved regions; there was nothing to "
                "reconstruct."
            )

        parts = [
            f"Examined {total_gaps} unresolved region(s) in {target.identifier} "
            f"({target.unknown_count} unknown bases, {target.completeness:.1%} complete) "
            f"over {iterations} reasoning iteration(s)."
        ]

        if resolved:
            bases = sum(
                len(item.reconstructed_sequence or "")
                for item in reconstructions
                if item.status is ReconstructionStatus.RECONSTRUCTED
            )
            confidences = [
                item.confidence
                for item in reconstructions
                if item.status is ReconstructionStatus.RECONSTRUCTED
            ]
            parts.append(
                f"Reconstructed {resolved} region(s), {bases} bases in total, with "
                f"confidence between {min(confidences):.0%} and {max(confidences):.0%}."
            )
        else:
            parts.append("No region could be reconstructed to the required confidence.")

        if by_status[ReconstructionStatus.LOW_CONFIDENCE]:
            parts.append(
                f"{by_status[ReconstructionStatus.LOW_CONFIDENCE]} region(s) produced a "
                "candidate below the confidence threshold; these are reported but should "
                "not be treated as resolved."
            )

        if by_status[ReconstructionStatus.UNRESOLVED]:
            parts.append(
                f"{by_status[ReconstructionStatus.UNRESOLVED]} region(s) had no usable "
                "reference evidence."
            )

        if skipped:
            parts.append(
                f"{len(skipped)} region(s) were not attempted: "
                + "; ".join(f"{gap_id} ({reason})" for gap_id, reason in skipped.items())
            )

        parts.append(
            "All reconstructed bases are inferences from homologous sequence, not "
            "observations, and should be verified before use."
        )
        return " ".join(parts)

    @staticmethod
    def remaining_unknowns(sequence: Sequence) -> int:
        return sequence.residues.count(UNKNOWN_BASE)
