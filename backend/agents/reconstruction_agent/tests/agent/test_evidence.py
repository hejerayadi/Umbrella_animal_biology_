"""Evidence accumulates, and no tool is allowed to stay silent.

Written after a measured failure: the graph produced an exact 45-base
reconstruction and reported it with no provenance and zero gap-spanning hits.
Nothing was wrong with the biology - the audit trail was simply assembled on a
different execution path from the one that gathered the evidence.
"""

from __future__ import annotations

import inspect

from reconstruction_agent.domain.models.evidence import (
    EvidenceBundle,
    EvidenceContribution,
    merge_evidence,
)
from reconstruction_agent.domain.models.result import AlignmentEvidence, HomologyEvidence
from reconstruction_agent.tools.alignment import AlignHomologsTool, AnalyzeAlignmentTool
from reconstruction_agent.tools.base import Tool
from reconstruction_agent.tools.candidate import GenerateCandidatesTool
from reconstruction_agent.tools.finalize import FinalizeResultTool
from reconstruction_agent.tools.homology import GetHomologSequencesTool, SearchHomologsTool
from reconstruction_agent.tools.sequence import GetAssemblyMetadataTool, GetSequenceContextTool


class TestMergingNeverLosesAMeasurement:
    def test_providers_accumulate_across_contributions(self) -> None:
        """A run that consulted a service does not stop having consulted it."""
        bundle = (
            EvidenceBundle()
            .merged_with(EvidenceContribution(providers=("NCBI",)))
            .merged_with(EvidenceContribution(providers=("NCBI BLAST",)))
            .merged_with(EvidenceContribution(providers=("EMBL-EBI MAFFT",)))
        )

        assert bundle.providers == ("NCBI", "NCBI BLAST", "EMBL-EBI MAFFT")

    def test_a_service_is_not_listed_twice(self) -> None:
        bundle = (
            EvidenceBundle()
            .merged_with(EvidenceContribution(providers=("NCBI",)))
            .merged_with(EvidenceContribution(providers=("NCBI",)))
        )

        assert bundle.providers == ("NCBI",)

    def test_an_empty_contribution_preserves_what_was_measured(self) -> None:
        """The exact regression: a later tool reporting nothing must not erase
        the search's hit counts."""
        measured = EvidenceContribution(
            homology=HomologyEvidence(hits_examined=50, gap_spanning_hits=50)
        )
        bundle = EvidenceBundle().merged_with(measured).merged_with(EvidenceContribution())

        assert bundle.homology is not None
        assert bundle.homology.gap_spanning_hits == 50

    def test_a_later_measurement_of_the_same_thing_wins(self) -> None:
        """A replan that re-searched has measured the same thing again, and the
        second measurement describes the evidence actually used."""
        bundle = (
            EvidenceBundle()
            .merged_with(EvidenceContribution(homology=HomologyEvidence(gap_spanning_hits=0)))
            .merged_with(EvidenceContribution(homology=HomologyEvidence(gap_spanning_hits=50)))
        )

        assert bundle.homology is not None
        assert bundle.homology.gap_spanning_hits == 50

    def test_the_reducer_merges_rather_than_replaces(self) -> None:
        """This is what the state channel actually calls."""
        left = EvidenceBundle(providers=("NCBI",), homology=HomologyEvidence(hits_examined=50))
        right = EvidenceBundle(providers=("EMBL-EBI MAFFT",), alignment=AlignmentEvidence())

        merged = merge_evidence(left, right)

        assert merged.providers == ("NCBI", "EMBL-EBI MAFFT")
        assert merged.homology is not None and merged.homology.hits_examined == 50
        assert merged.alignment is not None


class TestTheBundleBecomesTheReportedAuditTrail:
    def test_provenance_carries_the_services_and_the_rationale(self) -> None:
        bundle = EvidenceBundle(
            providers=("NCBI", "NCBI BLAST"),
            databases_searched=("scope@txid9632",),
            database_rationale="scoped to Ursidae (family)",
        )
        provenance = bundle.to_provenance(tool_calls=7)

        assert provenance.providers == ("NCBI", "NCBI BLAST")
        assert provenance.databases_searched == ("scope@txid9632",)
        assert "Ursidae" in provenance.database_rationale
        assert provenance.tool_calls == 7

    def test_an_empty_bundle_still_produces_a_usable_shape(self) -> None:
        """A refused gap reports evidence too - empty, not absent."""
        evidence = EvidenceBundle().to_gap_evidence()

        assert evidence.homology.gap_spanning_hits == 0
        assert evidence.evo2.consulted is False


class TestEveryToolContributes:
    """A tool that measures something and reports nothing is a silent hole in
    the audit trail. The contract is checked here rather than hoped for."""

    def test_every_evidence_producing_tool_overrides_contribute(self) -> None:
        producing = (
            GetSequenceContextTool,
            GetAssemblyMetadataTool,
            SearchHomologsTool,
            AlignHomologsTool,
            AnalyzeAlignmentTool,
            GenerateCandidatesTool,
        )
        missing = [tool.__name__ for tool in producing if tool.contribute is Tool.contribute]

        assert not missing, f"these tools contribute no evidence: {missing}"

    def test_contribute_is_part_of_the_base_contract(self) -> None:
        """So a new tool inherits a safe default instead of failing at runtime."""
        assert callable(Tool.contribute)
        assert "outcome" in inspect.signature(Tool.contribute).parameters

    def test_tools_with_nothing_to_measure_may_stay_silent(self) -> None:
        """Fetching sequences and finalising add no new measurement of their
        own; inheriting the empty default is correct, not an oversight."""
        assert GetHomologSequencesTool.contribute is Tool.contribute
        assert FinalizeResultTool.contribute is Tool.contribute
