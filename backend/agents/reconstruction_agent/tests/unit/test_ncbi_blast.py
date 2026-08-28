"""The NCBI BLAST provider: parsing its XML, and choosing its search scope.

Two providers now feed one domain model, so the conversions each side performs
have to agree. The identity test below is the one that matters: the two
services report the same quantity in different units, and getting it wrong
silently would score every candidate an order of magnitude off.
"""

from __future__ import annotations

import pytest
from tests.fakes import POLAR_BEAR_LINEAGE

from reconstruction_agent.config.settings import HomologySettings, NcbiBlastSettings
from reconstruction_agent.domain.enums import MoleculeType
from reconstruction_agent.domain.models.sequence import Gap, GapContext
from reconstruction_agent.domain.models.taxonomy import TargetProfile
from reconstruction_agent.integrations.ncbi_blast.parser import parse_hits
from reconstruction_agent.services.homology.homology_service import HomologyService

QUERY_LENGTH = 1000


def _xml(
    *,
    identity: int = 957,
    align_len: int = 1000,
    query_from: int = 1,
    query_to: int = 1000,
    hit_from: int = 4500,
    hit_to: int = 5499,
    accession: str = "AF303111",
    definition: str = "Ursus maritimus mitochondrion, complete genome",
) -> str:
    """A minimal BLAST XML document carrying one hit with one HSP."""
    return f"""<?xml version="1.0"?>
<BlastOutput>
  <BlastOutput_query-len>{QUERY_LENGTH}</BlastOutput_query-len>
  <BlastOutput_iterations>
    <Iteration>
      <Iteration_hits>
        <Hit>
          <Hit_num>1</Hit_num>
          <Hit_id>gi|12345|ref|{accession}.1|</Hit_id>
          <Hit_def>{definition}</Hit_def>
          <Hit_accession>{accession}</Hit_accession>
          <Hit_len>17017</Hit_len>
          <Hit_hsps>
            <Hsp>
              <Hsp_num>1</Hsp_num>
              <Hsp_bit-score>1848.0</Hsp_bit-score>
              <Hsp_evalue>0.0</Hsp_evalue>
              <Hsp_query-from>{query_from}</Hsp_query-from>
              <Hsp_query-to>{query_to}</Hsp_query-to>
              <Hsp_hit-from>{hit_from}</Hsp_hit-from>
              <Hsp_hit-to>{hit_to}</Hsp_hit-to>
              <Hsp_identity>{identity}</Hsp_identity>
              <Hsp_align-len>{align_len}</Hsp_align-len>
            </Hsp>
          </Hit_hsps>
        </Hit>
      </Iteration_hits>
    </Iteration>
  </BlastOutput_iterations>
</BlastOutput>"""


class TestParsingNcbiXml:
    def test_identity_is_a_ratio_not_a_percentage(self) -> None:
        """NCBI reports a count of matches; EBI reports a percentage.

        The same alignment is 957/1000 here and 95.7 there. Treating the count
        as a percentage would score a candidate ten times too high - or, since
        the domain model constrains identity to 0..1, fail validation loudly.
        Either way the two providers must agree on what the number means.
        """
        (hit,) = parse_hits(
            _xml(identity=957, align_len=1000), database_code="db", query_length=QUERY_LENGTH
        )

        assert hit.identity == pytest.approx(0.957)

    def test_coordinates_become_zero_based_half_open(self) -> None:
        (hit,) = parse_hits(_xml(), database_code="db", query_length=QUERY_LENGTH)

        assert (hit.query_start, hit.query_end) == (0, 1000)

    def test_a_minus_strand_hsp_is_normalised(self) -> None:
        (hit,) = parse_hits(
            _xml(hit_from=5499, hit_to=4500), database_code="db", query_length=QUERY_LENGTH
        )
        assert hit.subject_start < hit.subject_end

    def test_the_organism_is_read_the_same_way_as_for_ebi(self) -> None:
        """One parse rule for both providers, so relatedness is comparable."""
        (hit,) = parse_hits(_xml(), database_code="db", query_length=QUERY_LENGTH)

        assert hit.organism == "Ursus maritimus"
        assert hit.source_provider == "NCBI"

    def test_a_hit_crossing_the_junction_is_recognised(self) -> None:
        """The distinction between a homologue and a *usable* homologue."""
        (hit,) = parse_hits(_xml(), database_code="db", query_length=QUERY_LENGTH)

        assert hit.spans(junction=500) is True

    def test_an_empty_payload_means_no_hits_not_a_failure(self) -> None:
        """A completed search that found nothing returns no XML at all."""
        assert parse_hits("", database_code="db", query_length=QUERY_LENGTH) == ()

    def test_the_query_length_ncbi_reports_wins(self) -> None:
        """A truncated submission would otherwise score as full coverage."""
        (hit,) = parse_hits(_xml(align_len=500), database_code="db", query_length=10)

        assert hit.query_coverage == pytest.approx(0.5)

    def test_malformed_xml_is_rejected_rather_than_silently_empty(self) -> None:
        from reconstruction_agent.domain.exceptions import ExternalServiceError

        with pytest.raises(ExternalServiceError, match="not valid XML"):
            parse_hits("<not-xml", database_code="db", query_length=QUERY_LENGTH)


class TestSearchScope:
    """NCBI takes a tax id, so the scope is stated rather than inferred."""

    @staticmethod
    def _provider(**overrides: object) -> HomologyService:
        return HomologyService(
            blast=None,  # type: ignore[arg-type]  # not called by these tests
            taxonomy=None,  # type: ignore[arg-type]
            settings=NcbiBlastSettings(_env_file=None),
            homology=HomologySettings(_env_file=None, **overrides),  # type: ignore[arg-type]
        )

    @staticmethod
    def _polar_bear() -> TargetProfile:
        return TargetProfile(
            scientific_name="Ursus maritimus",
            tax_id=29073,
            taxonomy_lineage=POLAR_BEAR_LINEAGE,
            molecule_type=MoleculeType.MITOCHONDRION,
        )

    def test_scopes_are_taken_from_the_lineage_narrowest_first(self) -> None:
        """A close relative fills a gap accurately; a distant one does not."""
        scopes = self._provider()._scopes(self._polar_bear(), limit=3, exclude=frozenset())

        assert [scope.resolved_taxon.name for scope in scopes if scope.resolved_taxon] == [
            "Ursidae",
            "Carnivora",
            "Mammalia",
        ]

    def test_a_scope_names_a_tax_id_not_a_collection(self) -> None:
        """The whole point of this provider: nothing is inferred from a label."""
        scopes = self._provider()._scopes(self._polar_bear(), limit=1, exclude=frozenset())

        assert scopes[0].code.endswith("@txid9632")
        assert "Ursidae" in scopes[0].rationale

    def test_ranks_absent_from_the_lineage_are_skipped(self) -> None:
        """Not every lineage records every rank."""
        scopes = self._provider(ncbi_scope_ranks=("tribe", "family"))._scopes(
            self._polar_bear(), limit=3, exclude=frozenset()
        )

        assert len(scopes) == 1
        assert scopes[0].code.endswith("@txid9632")

    def test_an_organism_without_a_usable_rank_still_gets_scoped(self) -> None:
        """Better than an unrestricted search over the whole collection."""
        profile = TargetProfile(scientific_name="Unknown sp.", tax_id=12345)
        scopes = self._provider()._scopes(profile, limit=3, exclude=frozenset())

        assert len(scopes) == 1
        assert scopes[0].code.endswith("@txid12345")

    def test_an_organism_with_no_taxonomy_yields_no_scope(self) -> None:
        """Refusing beats searching the entire nucleotide collection blind."""
        scopes = self._provider()._scopes(
            TargetProfile(scientific_name="Unknown"), limit=3, exclude=frozenset()
        )

        assert scopes == ()

    def test_an_excluded_scope_is_not_proposed_again(self) -> None:
        provider = self._provider()
        first = provider._scopes(self._polar_bear(), limit=1, exclude=frozenset())
        second = provider._scopes(self._polar_bear(), limit=1, exclude=frozenset({first[0].code}))

        assert second[0].code != first[0].code


class TestTargetExclusion:
    """A record cannot be evidence about its own unresolved region."""

    @staticmethod
    def _context() -> GapContext:
        return GapContext(
            gap=Gap(gap_id="g1", start=10, end=20),
            source_accession="NC_003428.1",
            left_flank="ACGT",
            right_flank="ACGT",
        )

    def test_the_target_record_is_excluded_by_default(self) -> None:
        """Without this a ground-truth measurement is a lookup of the answer.

        The withheld bases are only withheld from the agent - the public record
        still has them, so leaving it in the search returns them at 100%.
        """
        provider = TestSearchScope._provider()

        assert provider._excluded_accession(self._context()) == "NC_003428.1"

    def test_exclusion_can_be_turned_off(self) -> None:
        provider = TestSearchScope._provider(exclude_target_accession=False)

        assert provider._excluded_accession(self._context()) is None
