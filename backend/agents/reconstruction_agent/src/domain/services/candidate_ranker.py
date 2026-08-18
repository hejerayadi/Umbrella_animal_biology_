"""Turn an alignment into proposed gap fillings, and pick between them.

This is where a reconstruction is actually produced: the reference bases in
the alignment columns that span the gap become the candidate sequence.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from contracts.output import EvidenceItem
from domain.models import Alignment, Candidate, GapContext, Reference
from domain.models.alignment import GAP_CHARACTER


@dataclass(frozen=True, slots=True)
class CandidateRanker:
    """Builds consensus candidates from an alignment and orders them.

    The consensus is per-column majority across the aligned references, with
    the agreement level recorded as `support`. A column where three references
    say `A` and one says `G` yields `A` with 0.75 support - the disagreement is
    not hidden, it is carried into the confidence score.
    """

    def build_consensus(
        self,
        context: GapContext,
        alignment: Alignment,
        references: list[Reference],
    ) -> Candidate | None:
        """The majority-vote candidate across every reference in the alignment.

        Returns None when the alignment does not locate the gap's columns:
        without them there is nothing to read out, and inventing a filling from
        the flanks alone would be fabrication.
        """
        if not alignment.spans_gap or not alignment.pairs:
            return None

        start = alignment.gap_column_start
        end = alignment.gap_column_end
        assert start is not None and end is not None  # guarded by spans_gap

        consensus_bases: list[str] = []
        column_supports: list[float] = []

        for column in range(start, end):
            votes = Counter(
                pair.reference_aligned[column]
                for pair in alignment.pairs
                if column < len(pair.reference_aligned)
            )
            # An alignment gap here means the references agree the target has
            # no base at this position - a deletion, not an unknown base. It is
            # a legitimate outcome, so it votes like any other symbol.
            if not votes:
                continue

            ranked = votes.most_common()
            base, count = ranked[0]
            runner_up = ranked[1][1] if len(ranked) > 1 else 0
            total = sum(votes.values())

            # Decisiveness, not raw share. The winner's share alone treats a
            # 3-vs-2 split as 0.6 - respectable-looking - when it is very nearly
            # a coin flip and carries almost no information. The margin over the
            # runner-up is 0 for a tie and 1 for unanimity, which is what the
            # confidence policy actually needs to know.
            #
            # This was not a theoretical concern: before the change, every
            # confidently-wrong case in scripts/tune_settings.py was a narrow
            # majority scoring ~0.75.
            column_supports.append((count - runner_up) / total)

            if base != GAP_CHARACTER:
                consensus_bases.append(base)

        if not consensus_bases:
            return None

        mean_support = sum(column_supports) / len(column_supports) if column_supports else 0.0
        by_accession = {reference.accession: reference for reference in references}

        return Candidate(
            gap_id=context.identifier,
            sequence="".join(consensus_bases),
            method="alignment_consensus",
            support=mean_support,
            supporting_references=[pair.reference_id for pair in alignment.pairs],
            evidence=[
                EvidenceItem(
                    source=alignment.tool,
                    reference_id=pair.reference_id,
                    organism=(
                        by_accession[pair.reference_id].organism
                        if pair.reference_id in by_accession
                        else None
                    ),
                    identity=pair.identity,
                    note="Aligned across the gap's flanking context.",
                )
                for pair in alignment.pairs
            ],
        )

    def rank(self, candidates: list[Candidate]) -> list[Candidate]:
        """Candidates best-first.

        Unscored candidates sort last rather than first: an unknown confidence
        is not evidence of a good one.
        """
        return sorted(
            candidates,
            key=lambda candidate: (
                -(candidate.confidence if candidate.confidence is not None else -1.0),
                -(candidate.support or 0.0),
                candidate.gap_id,
            ),
        )

    def best(self, candidates: list[Candidate]) -> Candidate | None:
        ranked = self.rank(candidates)
        return ranked[0] if ranked else None
