"""Turn an alignment into proposed gap fillings, and pick between them.

This is where a reconstruction is actually produced: the reference bases in
the alignment columns that span the gap become the candidate sequence.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass

from contracts.output import EvidenceItem
from domain.models import AlignedPair, Alignment, Candidate, GapContext, Reference
from domain.models.alignment import GAP_CHARACTER
from domain.policies.reference_quality import ReferenceQualityPolicy

#: What every reference's vote is worth before quality and proximity are
#: added. Kept well above zero so an unmeasured reference still counts.
_BASE_VOTE_WEIGHT = 1.0
_QUALITY_VOTE_WEIGHT = 1.0
_PROXIMITY_VOTE_WEIGHT = 0.5

#: A column whose winner beat the runner-up by less than this is contested:
#: the losing base is a hypothesis someone holds, not noise. Matches the
#: confidence policy's `ambiguous_support` - the region where the vote alone
#: cannot settle the answer is exactly where an independent check earns its
#: keep.
_CONTESTED_MARGIN = 0.25

#: How much of the alignment the two leading hypotheses must account for
#: before the disagreement is treated as two rival answers rather than as
#: ordinary divergence spread across many references.
_BIMODAL_COVERAGE = 0.6


@dataclass(frozen=True, slots=True)
class _ColumnVote:
    """One gap column's tally, before it is collapsed into a support figure."""

    winner: str
    runner_up: str
    winner_share: float
    runner_up_share: float
    runner_up_voters: tuple[str, ...]

    @property
    def margin(self) -> float:
        return self.winner_share - self.runner_up_share

    @property
    def contested(self) -> bool:
        """Whether the losing base is a real alternative rather than noise."""
        return self.runner_up_share > 0.0 and self.margin < _CONTESTED_MARGIN


@dataclass(frozen=True, slots=True)
class CandidateRanker:
    """Builds consensus candidates from an alignment and orders them.

    The consensus is per-column majority across the aligned references, with
    the agreement level recorded as `support`. A column where three references
    say `A` and one says `G` yields `A` with 0.75 support - the disagreement is
    not hidden, it is carried into the confidence score.

    Votes are weighted rather than counted: measured homology, phylogenetic
    proximity and annotation quality all change what a reference's opinion is
    worth. One-reference-one-vote let a distant, barely-matching pseudogene
    cancel a well-matched congener.
    """

    #: Injected so the evaluation suite can sweep it, and so a caller that
    #: knows better about its records can supply its own judgement.
    quality_policy: ReferenceQualityPolicy = ReferenceQualityPolicy()

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
        # How often each reference sided with the consensus. Depth is meant to
        # measure corroboration, so a reference that voted against the answer
        # must not count towards it - before this, dissent inflated confidence.
        agreements: Counter[str] = Counter()
        by_accession = {reference.accession: reference for reference in references}

        def weight_of(reference_id: str) -> float:
            """How much this reference's vote is worth.

            One-reference-one-vote let a 0.71-identity distant hit cancel a
            0.99-identity congener. Weighting by measured homology and
            phylogenetic proximity makes the consensus reflect the strength of
            the evidence rather than merely its quantity. Unmeasured references
            still vote - at the base weight, not zero, since an NCBI record
            with no alignment statistics is evidence too.
            """
            reference = by_accession.get(reference_id)
            if reference is None:
                return _BASE_VOTE_WEIGHT

            quality = reference.quality or 0.0
            proximity = reference.relatedness if reference.relatedness is not None else 0.5
            weight = _BASE_VOTE_WEIGHT + quality * _QUALITY_VOTE_WEIGHT + (
                proximity * _PROXIMITY_VOTE_WEIGHT
            )
            # A pseudogene aligns beautifully and then disagrees about exactly
            # the bases being reconstructed. Alignment statistics cannot see
            # that; the annotation can.
            return weight * self.quality_policy.penalty(reference)

        for column in range(start, end):
            votes: dict[str, float] = defaultdict(float)
            voters: dict[str, list[str]] = {}
            for pair in alignment.pairs:
                if column >= len(pair.reference_aligned):
                    continue
                base = pair.reference_aligned[column]
                votes[base] += weight_of(pair.reference_id)
                voters.setdefault(base, []).append(pair.reference_id)

            # An alignment gap here means the references agree the target has
            # no base at this position - a deletion, not an unknown base. It is
            # a legitimate outcome, so it votes like any other symbol.
            if not votes:
                continue

            ranked = sorted(votes.items(), key=lambda item: (-item[1], item[0]))
            base, count = ranked[0]
            runner_up = ranked[1][1] if len(ranked) > 1 else 0.0
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
            column_supports.append((count - runner_up) / total if total else 0.0)
            agreements.update(voters.get(base, []))

            if base != GAP_CHARACTER:
                consensus_bases.append(base)

        if not consensus_bases:
            return None

        mean_support = sum(column_supports) / len(column_supports) if column_supports else 0.0
        columns = len(column_supports) or 1
        # A reference supports the candidate when it sided with the consensus
        # for most of the gap, not merely when it was present in the alignment.
        supporting = [
            reference_id
            for reference_id, agreed in agreements.items()
            if agreed >= columns / 2
        ]

        return Candidate(
            gap_id=context.identifier,
            sequence="".join(consensus_bases),
            method="alignment_consensus",
            support=mean_support,
            supporting_references=supporting or [p.reference_id for p in alignment.pairs],
            evidence=self._evidence(alignment, by_accession),
        )

    def build_alternatives(
        self,
        context: GapContext,
        alignment: Alignment,
        references: list[Reference],
    ) -> list[Candidate]:
        """Every fill the evidence genuinely leaves open, best-supported first.

        A contested column has a real second answer: the references that lost
        the vote are not noise, they are a competing hypothesis about what the
        target carries. Collapsing that to a single plurality string throws the
        disagreement away at exactly the moment it matters, and leaves nothing
        for an independent check to arbitrate between.

        Returns one candidate when the consensus is clean, and two when it is
        contested - the plurality fill and the runner-up fill. Never more: a
        third-choice base at a coin-flip column is not a hypothesis anyone holds,
        and each extra candidate costs an Evo 2 call to judge.
        """
        primary = self.build_consensus(context, alignment, references)
        if primary is None:
            return []

        columns = self._columns(alignment, alignment.pairs, references)
        contested = [column for column in columns if column.contested]
        if not contested:
            return [primary]

        # When the vote is contested, the hypotheses are formed by grouping the
        # references on the fill they each propose - not by picking a winner
        # column by column. Per-column selection builds a chimera: two
        # references split at every position yield neither of their sequences
        # but a positional mixture of both, a fill no homologue carries and no
        # model can sensibly judge. Verified against live Evo 2, which was being
        # asked to choose between two sequences that did not exist.
        groups = self._fill_groups(alignment, references)
        if len(groups) < 2:
            return [primary]

        covered = sum(len(pairs) for _, pairs in groups[:2])
        if covered < len(alignment.pairs) * _BIMODAL_COVERAGE:
            # Many references each differing slightly - real divergence rather
            # than two rival answers. Grouping would fragment into singletons
            # and elevate one arbitrary sequence, so the per-column consensus
            # remains the better summary.
            return [primary]

        (first_weight, first_pairs), (second_weight, second_pairs) = groups[:2]
        total = first_weight + second_weight or 1.0

        candidates = [
            self._consensus_over(
                context,
                alignment,
                pairs,
                references,
                method=method,
                # Standing in the whole vote, not agreement within the group:
                # a lone dissenter agrees with itself perfectly and would
                # otherwise outscore the majority it lost to.
                support=max(0.0, (weight - other) / total),
            )
            for weight, pairs, other, method in (
                (first_weight, first_pairs, second_weight, "alignment_consensus"),
                (second_weight, second_pairs, first_weight, "dissenting_consensus"),
            )
        ]

        alive = [candidate for candidate in candidates if candidate is not None]
        if len(alive) < 2 or alive[0].sequence == alive[1].sequence:
            return [primary]

        return alive


    def _fill_groups(
        self, alignment: Alignment, references: list[Reference]
    ) -> list[tuple[float, list[AlignedPair]]]:
        """The references grouped by the fill they propose, heaviest group first.

        Weight rather than count, so a group of well-matched close relatives
        outranks a larger group of distant poor matches - the same judgement the
        per-column vote makes, applied to whole hypotheses.
        """
        start, end = alignment.gap_column_start, alignment.gap_column_end
        if start is None or end is None:
            return []

        by_accession = {reference.accession: reference for reference in references}
        grouped: dict[str, list[AlignedPair]] = defaultdict(list)

        for pair in alignment.pairs:
            fill = pair.reference_aligned[start:end]
            if not fill.strip(GAP_CHARACTER):
                # Contributes nothing across the gap; not a hypothesis.
                continue
            grouped[fill].append(pair)

        weighted = [
            (sum(self._weight(pair.reference_id, by_accession) for pair in pairs), pairs)
            for pairs in grouped.values()
        ]
        # Sorted on the fill as a tie-break so equally-weighted groups keep a
        # stable order across runs.
        return sorted(
            weighted,
            key=lambda item: (-item[0], item[1][0].reference_aligned[start:end]),
        )

    @staticmethod
    def _dissenting_pairs(
        alignment: Alignment, columns: list[_ColumnVote]
    ) -> list[AlignedPair]:
        """The references that lost the contested columns, as a group.

        A reference dissents when it voted against the winner in most of the
        columns that were close. One that merely differs at a single position is
        not proposing a rival reconstruction.
        """
        contested = [
            (index, column)
            for index, column in enumerate(columns)
            if column.contested
        ]
        if not contested:
            return []

        start = alignment.gap_column_start or 0
        dissenting: list[AlignedPair] = []

        for pair in alignment.pairs:
            against = sum(
                1
                for index, column in contested
                if start + index < len(pair.reference_aligned)
                and pair.reference_aligned[start + index] != column.winner
            )
            if against > len(contested) / 2:
                dissenting.append(pair)

        return dissenting

    def _columns(
        self,
        alignment: Alignment,
        pairs: list[AlignedPair],
        references: list[Reference],
    ) -> list[_ColumnVote]:
        """The per-column tally, kept rather than collapsed to a mean.

        `build_consensus` reduces this to a single support figure. Alternatives
        and per-base reporting both need the columns themselves. `pairs` is
        passed rather than read off the alignment so a subset - the dissenting
        references - can be tallied on its own.
        """
        start, end = alignment.gap_column_start, alignment.gap_column_end
        if start is None or end is None:
            return []

        by_accession = {reference.accession: reference for reference in references}
        columns: list[_ColumnVote] = []

        for column in range(start, end):
            votes: dict[str, float] = defaultdict(float)
            voters: dict[str, list[str]] = {}
            for pair in pairs:
                if column >= len(pair.reference_aligned):
                    continue
                base = pair.reference_aligned[column]
                votes[base] += self._weight(pair.reference_id, by_accession)
                voters.setdefault(base, []).append(pair.reference_id)

            if not votes:
                continue

            ranked = sorted(votes.items(), key=lambda item: (-item[1], item[0]))
            total = sum(votes.values()) or 1.0
            winner, winner_votes = ranked[0]
            runner_up, runner_up_votes = (
                ranked[1] if len(ranked) > 1 else (GAP_CHARACTER, 0.0)
            )

            columns.append(
                _ColumnVote(
                    winner=winner,
                    runner_up=runner_up,
                    winner_share=winner_votes / total,
                    runner_up_share=runner_up_votes / total,
                    runner_up_voters=tuple(voters.get(runner_up, ())),
                )
            )

        return columns


    def _consensus_over(
        self,
        context: GapContext,
        alignment: Alignment,
        pairs: list[AlignedPair],
        references: list[Reference],
        *,
        method: str,
        support: float | None = None,
    ) -> Candidate | None:
        """The fill a given subset of the aligned references agrees on.

        `support` is passed in for a subset, because agreement measured *within*
        the subset is the same self-agreement degeneracy the depth factor
        guards against: one dissenting reference agrees with itself perfectly
        and would score 1.0, making a lone objector look more decisive than the
        majority it lost to. What the subset's standing actually is has to be
        read off the whole vote.
        """
        columns = self._columns(alignment, pairs, references)
        if not columns:
            return None

        bases = [column.winner for column in columns if column.winner != GAP_CHARACTER]
        if not bases:
            return None

        if support is None:
            support = sum(column.margin for column in columns) / len(columns)
        by_accession = {reference.accession: reference for reference in references}

        return Candidate(
            gap_id=context.identifier,
            sequence="".join(bases),
            method=method,
            support=max(0.0, support),
            supporting_references=[pair.reference_id for pair in pairs],
            evidence=self._evidence(alignment, by_accession),
        )

    def _weight(self, reference_id: str, by_accession: dict[str, Reference]) -> float:
        """How much one reference's vote is worth. See `build_consensus`."""
        reference = by_accession.get(reference_id)
        if reference is None:
            return _BASE_VOTE_WEIGHT

        quality = reference.quality or 0.0
        proximity = reference.relatedness if reference.relatedness is not None else 0.5
        weight = _BASE_VOTE_WEIGHT + quality * _QUALITY_VOTE_WEIGHT + (
            proximity * _PROXIMITY_VOTE_WEIGHT
        )
        return weight * self.quality_policy.penalty(reference)

    @staticmethod
    def _evidence(
        alignment: Alignment, by_accession: dict[str, Reference]
    ) -> list[EvidenceItem]:
        return [
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
        ]

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
