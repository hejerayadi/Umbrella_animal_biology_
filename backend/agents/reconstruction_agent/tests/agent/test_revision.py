"""Diagnosing why a gap failed, and planning something different because of it.

A REVISE that carries no reason leaves the planner with nowhere to go but the
call it just made, so the second iteration produces the same evidence and the
same objection at twice the cost. These tests pin both halves: that the
diagnosis names the first broken link in the evidence chain, and that the plan
actually changes because of it.
"""
from __future__ import annotations

from agent.planning.planner import Planner
from agent.reasoning.revision import RevisionReason, diagnose, diagnose_all
from agent.state.state import initial_state
from domain.models import Alignment, Candidate, Gap, GapContext, Reference, Sequence

FLANK = "ACGTTGCA" * 10


def make_state(**overrides: object) -> dict:
    context = GapContext(
        gap=Gap("gap_1", 80, 110), left_flank=FLANK, right_flank=FLANK
    )
    state = initial_state(
        run_id="run-1",
        trace_id="trace-1",
        instruction="Reconstruct.",
        target=Sequence.parse("seq", FLANK + "N" * 30 + FLANK),
        organism="Testus organismus",
        requested_organisms=[],
        max_iterations=6,
        max_slices=4,
    )
    state["gap_contexts"] = [context]
    state.update(overrides)  # type: ignore[typeddict-item]
    return dict(state)


def reference(accession: str, **kwargs: object) -> Reference:
    return Reference(accession=accession, **kwargs)  # type: ignore[arg-type]


def spanning_alignment() -> Alignment:
    return Alignment(gap_id="gap_1", pairs=[], gap_column_start=80, gap_column_end=110)


class TestDiagnosis:
    def test_no_references_at_all(self) -> None:
        assert diagnose(make_state(), "gap_1") is RevisionReason.NO_BLAST_HITS

    def test_hits_that_cannot_be_aligned(self) -> None:
        """The historical failure: BLAST metadata with no residues behind it."""
        state = make_state(references={"gap_1": [reference("REF_1", identity=0.99)]})

        assert diagnose(state, "gap_1") is RevisionReason.BAD_REFERENCES

    def test_alignment_that_does_not_locate_the_gap(self) -> None:
        state = make_state(
            references={"gap_1": [reference("REF_1", residues="ACGT", identity=0.9)]},
            alignments={"gap_1": Alignment(gap_id="gap_1", pairs=[])},
        )

        assert diagnose(state, "gap_1") is RevisionReason.WEAK_ALIGNMENT

    def test_references_too_divergent_to_read_a_fill_from(self) -> None:
        state = make_state(
            references={"gap_1": [reference("REF_1", residues="ACGT", identity=0.4)]},
            alignments={"gap_1": spanning_alignment()},
        )

        assert diagnose(state, "gap_1") is RevisionReason.BAD_REFERENCES

    def test_an_evenly_split_consensus(self) -> None:
        """A coin flip, and the largest source of confidently-wrong answers."""
        state = make_state(
            references={"gap_1": [reference("REF_1", residues="ACGT", identity=0.95)]},
            alignments={"gap_1": spanning_alignment()},
            candidates={"gap_1": [Candidate(gap_id="gap_1", sequence="ACGT", support=0.05)]},
        )

        assert diagnose(state, "gap_1") is RevisionReason.AMBIGUOUS_CONSENSUS

    def test_only_distant_relatives(self) -> None:
        state = make_state(
            references={
                "gap_1": [
                    reference("REF_1", residues="ACGT", identity=0.95, relatedness=0.3)
                ]
            },
            alignments={"gap_1": spanning_alignment()},
            candidates={"gap_1": [Candidate(gap_id="gap_1", sequence="ACGT", support=0.9)]},
        )

        assert diagnose(state, "gap_1") is RevisionReason.INSUFFICIENT_PHYLOGENETIC_SUPPORT

    def test_coherent_evidence_has_nothing_to_diagnose(self) -> None:
        """Scoring below the threshold is not itself a broken link."""
        state = make_state(
            references={
                "gap_1": [
                    reference("REF_1", residues="ACGT", identity=0.95, relatedness=0.9)
                ]
            },
            alignments={"gap_1": spanning_alignment()},
            candidates={"gap_1": [Candidate(gap_id="gap_1", sequence="ACGT", support=0.9)]},
        )

        assert diagnose(state, "gap_1") is None

    def test_the_earliest_broken_link_wins(self) -> None:
        """Fixing a later link is pointless while an earlier one is still broken."""
        state = make_state(
            references={"gap_1": [reference("REF_1", identity=0.1, relatedness=0.1)]},
            candidates={"gap_1": [Candidate(gap_id="gap_1", sequence="A", support=0.0)]},
        )

        assert diagnose(state, "gap_1") is RevisionReason.BAD_REFERENCES

    def test_diagnoses_survive_a_checkpoint_as_plain_strings(self) -> None:
        reasons = diagnose_all(make_state(), {"gap_1"})

        assert reasons == {"gap_1": "no_blast_hits"}
        assert all(isinstance(value, str) for value in reasons.values())


class TestPlanChangesStrategy:
    """The point of a diagnosis: the next move must not be the last one again."""

    def plan_for(self, reason: str, **overrides: object) -> list[str]:
        state = make_state(revision_reasons={"gap_1": reason}, **overrides)
        return [step.tool for step in Planner(None, []).deterministic_plan(state)]  # type: ignore[arg-type]

    def test_unalignable_hits_send_it_to_fetch_sequences_by_name(self) -> None:
        tools = self.plan_for(
            "bad_references", references={"gap_1": [reference("REF_1", identity=0.9)]}
        )

        assert "ncbi_search" in tools
        assert "blast_search" not in tools, "repeating the search returns the same metadata"

    def test_a_weak_alignment_realigns_rather_than_researching(self) -> None:
        tools = self.plan_for(
            "weak_alignment",
            references={"gap_1": [reference("REF_1", residues="ACGT")]},
        )

        assert tools[0] == "mafft_align"

    def test_an_ambiguous_consensus_gathers_more_independent_evidence(self) -> None:
        tools = self.plan_for(
            "ambiguous_consensus",
            references={"gap_1": [reference("REF_1", residues="ACGT")]},
            alignments={"gap_1": spanning_alignment()},
        )

        assert "blast_search" in tools

    def test_distant_relatives_are_ranked_before_being_read_from(self) -> None:
        tools = self.plan_for(
            "insufficient_phylogenetic_support",
            references={"gap_1": [reference("REF_1", residues="ACGT", relatedness=0.3)]},
            alignments={"gap_1": spanning_alignment()},
        )

        assert "evolutionary_context" in tools

    def test_an_unknown_reason_falls_back_to_the_standard_pipeline(self) -> None:
        """A diagnosis the planner does not recognise must not stall the gap."""
        tools = self.plan_for("something_new")

        assert tools == ["blast_search"]
