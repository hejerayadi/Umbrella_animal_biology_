"""What makes an assembly worth escalating to the Reconstruction Agent.

The decision lives in `get_genome_metadata_node` and has two independent
halves: the assembly's *level*, and the *share of it that is unresolved*.
Neither is sufficient alone, and the second one is easy to get wrong in a way
no unit test on a synthetic number would catch - which is what these fixtures
are for.

The six assemblies below are real, and their numbers were read from NCBI's
own assembly stats rather than invented. They are frozen here as fixtures so
the test is offline and deterministic; the point is not to re-measure NCBI but
to pin the *classification* against a spread that actually occurs in nature:

    total_length - ungapped_length, as a share of total_length

    Complete, finished       0.0000%   escalating these would be nonsense
    Chromosome, near-perfect 0.0004%   <- the case a bare `> 0` gets wrong
    Scaffold                 0.4040%
    Chromosome, gappy        2.6977%
    Chromosome, gappier      4.8996%

The tiger is the fixture that earns its place. Ten kilobases of N in 2.4 Gb is
an assembly with a few uncallable bases, not one with unresolved regions worth
a reconstruction run - and it is the control case in the handoff audit
precisely because it must complete without escalating. A first cut of the
gap-count trigger fired on any count above zero and escalated it.
"""

from __future__ import annotations

import asyncio
from unittest.mock import patch

import pytest

from genome_agent.workflows.nodes import genome_data_nodes as gdn
from genome_agent.workflows.state import GenomeAgentState

# (assembly, organism, level, genome_size_bp, gap_bases_bp, should_escalate)
ASSEMBLIES = [
    ("GCF_009914755.1", "H. sapiens T2T", "Complete Genome", 3_117_275_501, 0, False),
    ("GCF_000002985.6", "C. elegans", "Complete Genome", 100_286_401, 0, False),
    ("GCF_018350195.1", "P. tigris", "Chromosome", 2_408_695_688, 10_100, False),
    ("GCF_017311325.1", "U. maritimus", "Scaffold", 2_330_485_441, 9_414_993, True),
    ("GCF_000001635.27", "M. musculus", "Chromosome", 2_728_222_451, 73_600_614, True),
    ("GCF_000001405.40", "H. sapiens GRCh38", "Chromosome", 3_298_430_636, 161_611_139, True),
]


def _run(assembly_id: str, metadata: dict) -> dict | None:
    """The node's escalation decision for one assembly, with NCBI stubbed."""
    state = GenomeAgentState(user_question="q", assembly_id=assembly_id, needs_metadata=True)

    async def _fake(_assembly_id):
        return metadata

    with patch.object(gdn, "get_genome_metadata", new=_fake):
        return asyncio.run(gdn.get_genome_metadata_node(state))["reconstruction_need"]


def _metadata(level: str, size: int | None, gaps: int | None) -> dict:
    return {
        "assembly_level": level,
        "genome_size_bp": size,
        "chromosome_count": None,
        "karyotype": None,
        "gap_bases_bp": gaps,
    }


@pytest.mark.parametrize(
    "assembly_id,organism,level,size,gaps,should_escalate",
    ASSEMBLIES,
    ids=[row[1] for row in ASSEMBLIES],
)
def test_real_assemblies_are_classified_correctly(
    assembly_id, organism, level, size, gaps, should_escalate
):
    need = _run(assembly_id, _metadata(level, size, gaps))
    assert (need is not None) is should_escalate, (
        f"{organism} ({level}, {gaps:,} of {size:,} bp = {gaps / size:.4%}) "
        f"should{'' if should_escalate else ' not'} escalate"
    )


class TestTheTwoHalvesOfTheDecision:
    def test_an_incomplete_level_escalates_even_with_no_gap_count(self):
        """A Scaffold is escalated on its level alone. Losing that when the
        gap count is unavailable would silently narrow the original trigger."""
        assert _run("GCF_x.1", _metadata("Scaffold", 2_000_000, None)) is not None

    def test_a_finished_level_needs_a_gap_share_over_the_floor(self):
        just_under = int(2_000_000 * gdn._MIN_GAP_FRACTION) - 1
        just_over = int(2_000_000 * gdn._MIN_GAP_FRACTION) + 1
        assert _run("GCF_x.1", _metadata("Chromosome", 2_000_000, just_under)) is None
        assert _run("GCF_x.1", _metadata("Chromosome", 2_000_000, just_over)) is not None

    def test_an_unmeasured_gap_count_is_not_read_as_zero_or_as_gappy(self):
        """None means "we did not find out". It must not escalate by itself,
        and it must not suppress the level check either."""
        assert _run("GCF_x.1", _metadata("Chromosome", 2_000_000, None)) is None
        assert _run("GCF_x.1", _metadata("Scaffold", 2_000_000, None)) is not None

    def test_a_zero_genome_size_cannot_produce_a_division_error(self):
        """The fraction needs a non-zero denominator. A zero-length assembly is
        nonsense NCBI has served before; it must classify, not raise."""
        assert _run("GCF_x.1", _metadata("Chromosome", 0, 50_000)) is None

    def test_a_missing_genome_size_stops_before_the_decision_is_reached(self):
        """Documented rather than asserted-around: a null `genome_size_bp` is
        the node's existing signal that NCBI returned nothing usable, and it
        returns an error without a `reconstruction_need` key at all. There is
        no escalation decision to make on metadata that does not exist - but
        the absence of the key is a contract a caller has to know about, and
        it is easy to mistake for "decided not to escalate"."""
        state = GenomeAgentState(user_question="q", assembly_id="GCF_x.1", needs_metadata=True)

        async def _fake(_assembly_id):
            return _metadata("Scaffold", None, 50_000)

        with patch.object(gdn, "get_genome_metadata", new=_fake):
            out = asyncio.run(gdn.get_genome_metadata_node(state))

        assert "reconstruction_need" not in out
        assert out["metadata"] is None
        assert len(out["errors"]) == 1


class TestTheEscalationCarriesItsEvidence:
    def test_the_payload_states_both_the_count_and_the_share(self):
        """The consumer is told what escalated it. The count alone does not
        explain the decision - the share is what was judged."""
        need = _run("GCF_000001635.27", _metadata("Chromosome", 2_728_222_451, 73_600_614))
        assert need["gap_bases_bp"] == 73_600_614
        assert need["gap_fraction"] == pytest.approx(0.026977, abs=1e-6)

    def test_the_reason_names_the_level_that_hid_the_gaps(self):
        need = _run("GCF_000001635.27", _metadata("Chromosome", 2_728_222_451, 73_600_614))
        prompt = need["prompt_to_target_agent"]
        assert "Chromosome" in prompt
        assert "73,600,614" in prompt

    def test_a_scaffold_reason_does_not_claim_a_gap_count_it_did_not_use(self):
        need = _run("GCF_017311325.1", _metadata("Scaffold", 2_330_485_441, 9_414_993))
        assert "Scaffold" in need["prompt_to_target_agent"]
