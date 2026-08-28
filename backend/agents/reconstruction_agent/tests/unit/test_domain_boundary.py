"""The separation between biology and provider vocabulary.

The first test here is the one that matters most. It is a structural assertion
rather than a behavioural one: it fails if anybody ever puts a database code,
an ENA division, or any other provider concept back onto the domain model.

That coupling is what this rewrite removed. When a taxonomic fact and a
provider catalogue layout live in the same field, a disagreement between them
is invisible - the profile looks correct while the search runs against a
collection that cannot contain the answer.
"""

from __future__ import annotations

import pytest
from tests.fakes import MAMMALIA, POLAR_BEAR_LINEAGE, VERTEBRATA

from reconstruction_agent.domain.enums import MoleculeType, ReconstructionStatus
from reconstruction_agent.domain.models.result import GapReconstruction, ReconstructionResult
from reconstruction_agent.domain.models.sequence import Gap, GapContext, SequenceRecord
from reconstruction_agent.domain.models.taxonomy import TargetProfile


class TestTargetProfileCarriesBiologyOnly:
    def test_the_profile_has_exactly_the_biological_fields(self) -> None:
        """No provider vocabulary may appear on a domain model.

        Anything the domain knows must remain true if EMBL-EBI is replaced
        tomorrow.
        """
        assert set(TargetProfile.model_fields) == {
            "scientific_name",
            "tax_id",
            "taxonomy_lineage",
            "taxonomy_ranks",
            "molecule_type",
        }

    @pytest.mark.parametrize(
        "forbidden", ["ena_division", "blast_database", "database", "division"]
    )
    def test_no_provider_field_has_crept_back(self, forbidden: str) -> None:
        assert forbidden not in TargetProfile.model_fields


class TestContainment:
    """The comparison the whole database-selection design rests on."""

    def test_specificity_is_read_from_the_lineage(self) -> None:
        """Nothing states that mammals are vertebrates - the lineage does.

        A deeper position means a closer clade, which is what lets one
        collection be preferred over another with no vocabulary of our own.
        """
        profile = TargetProfile(
            scientific_name="Ursus maritimus",
            tax_id=29073,
            taxonomy_lineage=POLAR_BEAR_LINEAGE,
        )

        mammal_depth = profile.depth_of(MAMMALIA.tax_id)
        vertebrate_depth = profile.depth_of(VERTEBRATA.tax_id)

        assert mammal_depth is not None and vertebrate_depth is not None
        assert mammal_depth > vertebrate_depth

    def test_an_unrelated_clade_has_no_depth(self) -> None:
        profile = TargetProfile(
            scientific_name="Ursus maritimus",
            tax_id=29073,
            taxonomy_lineage=POLAR_BEAR_LINEAGE,
        )
        assert profile.depth_of(8782) is None  # Aves
        assert profile.contains(8782) is False


class TestGapCoordinates:
    def test_a_gap_slices_the_record_it_came_from(self) -> None:
        """Half-open coordinates, so the gap is a plain Python slice."""
        record = SequenceRecord(accession="X", residues="AAAA" + "NNNNN" + "CCCC")
        gap = Gap(gap_id="g1", start=4, end=9)

        assert gap.length == 5
        assert record.residues[gap.start : gap.end] == "NNNNN"
        assert record.left_flank_of(gap, 4) == "AAAA"
        assert record.right_flank_of(gap, 4) == "CCCC"

    def test_an_inverted_gap_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="must exceed start"):
            Gap(gap_id="g1", start=100, end=50)

    def test_the_query_joins_the_flanks_so_the_junction_is_the_gap(self) -> None:
        """The missing bases are omitted rather than sent as ambiguity.

        A run of N carries no information and degrades the search; what is
        wanted is a reference matching both flanks, because that is what can
        span the hole between them.
        """
        context = GapContext(
            gap=Gap(gap_id="g1", start=4, end=9), left_flank="AAAA", right_flank="CCCC"
        )

        assert context.query_sequence() == "AAAACCCC"
        assert len(context.left_flank) == 4

    def test_a_gap_missing_a_flank_is_not_usable(self) -> None:
        """One flank can place an edge of the fill but cannot bound it."""
        context = GapContext(gap=Gap(gap_id="g1", start=4, end=9), left_flank="AAAA")
        assert context.has_usable_flanks is False


class TestOverallStatus:
    """A partly successful run is a result, not a failure."""

    @staticmethod
    def _gap(index: int, resolved: bool) -> GapReconstruction:
        from reconstruction_agent.domain.enums import GapStatus, UnresolvedReason
        from reconstruction_agent.domain.models.candidate import Candidate

        gap = Gap(gap_id=f"g{index}", start=index * 100, end=index * 100 + 10)
        if resolved:
            return GapReconstruction(
                gap=gap,
                status=GapStatus.RESOLVED,
                selected_candidate=Candidate(
                    candidate_id=f"c{index}", gap_id=gap.gap_id, sequence="ACGT"
                ),
            )
        return GapReconstruction(
            gap=gap,
            status=GapStatus.UNRESOLVED,
            unresolved_reason=UnresolvedReason.INSUFFICIENT_GAP_SPANNING_HOMOLOGS,
        )

    def test_eight_of_ten_is_partially_completed_not_failed(self) -> None:
        """Collapsing this into FAILED would discard eight reconstructions."""
        result = ReconstructionResult(
            request_id="r1",
            reconstructions=tuple(self._gap(index, resolved=index < 8) for index in range(10)),
        )

        assert result.status is ReconstructionStatus.PARTIALLY_COMPLETED
        assert (result.resolved_gaps, result.unresolved_gaps) == (8, 2)

    def test_all_resolved_is_completed(self) -> None:
        result = ReconstructionResult(
            request_id="r1", reconstructions=(self._gap(0, resolved=True),)
        )
        assert result.status is ReconstructionStatus.COMPLETED

    def test_none_resolved_is_unresolved_not_failed(self) -> None:
        """Refusing to invent a sequence is a valid scientific answer."""
        result = ReconstructionResult(
            request_id="r1", reconstructions=(self._gap(0, resolved=False),)
        )
        assert result.status is ReconstructionStatus.UNRESOLVED


class TestMoleculeType:
    def test_organelle_types_are_recognised(self) -> None:
        assert MoleculeType.MITOCHONDRION.is_organelle is True
        assert MoleculeType.GENOMIC_DNA.is_organelle is False
