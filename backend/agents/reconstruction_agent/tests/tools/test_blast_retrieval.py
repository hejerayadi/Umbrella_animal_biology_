"""The BLAST tool retrieving the sequence behind each hit.

This is the seam the whole evidence path used to break at: BLAST answers with
alignment statistics, never with the missing segment, so references arrived
with no residues and `ToolSelector._mafft` filtered every one of them out. The
default `blast_search -> mafft_align` plan could not align anything at all.

Nothing here talks to a network: the EBI payload and the NCBI response are both
supplied as fakes, so what is under test is the wiring between them.
"""
from __future__ import annotations

import json

import pytest

from domain.exceptions import ExternalServiceError
from tools.blast.schemas import BlastSearchInput
from tools.blast.tool import BlastSearchTool

pytestmark = pytest.mark.asyncio


def payload_with(*hits: dict[str, object]) -> str:
    return json.dumps({"hits": list(hits)})


BRACKETING_HIT = {
    "hit_acc": "REF_1",
    "hit_os": "Loxodonta africana",
    "hit_hsps": [
        {"hsp_identity": 95.0, "hsp_expect": 1e-40, "hsp_hit_from": 100, "hsp_hit_to": 200},
        {"hsp_identity": 94.0, "hsp_expect": 1e-38, "hsp_hit_from": 260, "hsp_hit_to": 360},
    ],
}


class FakeBlastClient:
    def __init__(self, raw: str) -> None:
        self._raw = raw
        self.submitted: list[str] = []

    async def submit(self, sequence: str, **_: object) -> str:
        self.submitted.append(sequence)
        return "job-1"

    async def result(self, job_id: str, **_: object) -> str:
        return self._raw


class FakeNCBIClient:
    """Records what regions were asked for, and answers with FASTA."""

    def __init__(self, residues: str = "ACGTACGTAC", fail: bool = False) -> None:
        self._residues = residues
        self._fail = fail
        self.requests: list[tuple[str, int, int, int]] = []

    async def fetch_region(
        self, identifier: str, start: int, stop: int, *, database: str = "nuccore", strand: int = 1
    ) -> str:
        self.requests.append((identifier, start, stop, strand))
        if self._fail:
            raise ExternalServiceError("ncbi", "simulated outage")
        return f">{identifier} region\n{self._residues}\n"


def make_tool(raw: str, ncbi: FakeNCBIClient | None = None) -> BlastSearchTool:
    return BlastSearchTool(FakeBlastClient(raw), ncbi)  # type: ignore[arg-type]


class TestSequenceRetrieval:
    async def test_bracketed_hits_come_back_alignable(self) -> None:
        ncbi = FakeNCBIClient()
        tool = make_tool(payload_with(BRACKETING_HIT), ncbi)

        output = await tool.run(
            BlastSearchInput(sequence="ACGT" * 30, gap_id="gap_1", gap_length=50)
        )

        assert output.succeeded
        assert output.references[0].has_sequence, (
            "a hit without residues is discarded before it ever reaches the aligner"
        )
        assert output.references[0].residues == "ACGTACGTAC"

    async def test_the_fetched_region_covers_the_missing_segment(self) -> None:
        """Widened by the gap, because the fill is in neither HSP."""
        ncbi = FakeNCBIClient()
        tool = make_tool(payload_with(BRACKETING_HIT), ncbi)

        await tool.run(BlastSearchInput(sequence="ACGT" * 30, gap_id="gap_1", gap_length=50))

        accession, start, stop, _ = ncbi.requests[0]
        assert accession == "REF_1"
        assert start == 50 and stop == 410

    async def test_minus_strand_is_requested_from_ncbi_not_flipped_here(self) -> None:
        hit = {
            "hit_acc": "REF_1",
            "hit_hsps": [
                {"hsp_hit_from": 100, "hsp_hit_to": 200, "hsp_strand": "plus/minus"},
                {"hsp_hit_from": 260, "hsp_hit_to": 360, "hsp_strand": "plus/minus"},
            ],
        }
        ncbi = FakeNCBIClient()
        tool = make_tool(payload_with(hit), ncbi)

        await tool.run(BlastSearchInput(sequence="ACGT" * 30, gap_length=10))

        assert ncbi.requests[0][3] == -1

    async def test_reports_how_many_hits_became_alignable(self) -> None:
        ncbi = FakeNCBIClient()
        tool = make_tool(payload_with(BRACKETING_HIT, {**BRACKETING_HIT, "hit_acc": "REF_2"}), ncbi)

        output = await tool.run(BlastSearchInput(sequence="ACGT" * 30, gap_length=10))

        assert output.diagnostics["blast_hits_total"] == 2
        assert output.diagnostics["blast_hits_with_sequence"] == 2

    async def test_a_failed_fetch_loses_only_that_reference(self) -> None:
        """An unreachable accession must not discard the rest of the search."""
        ncbi = FakeNCBIClient(fail=True)
        tool = make_tool(payload_with(BRACKETING_HIT), ncbi)

        output = await tool.run(BlastSearchInput(sequence="ACGT" * 30, gap_length=10))

        assert output.succeeded
        assert output.total_hits == 1
        assert output.diagnostics["blast_hits_with_sequence"] == 0

    async def test_runs_without_an_ncbi_client(self) -> None:
        """Degraded rather than broken: the hits are still reported."""
        tool = make_tool(payload_with(BRACKETING_HIT), None)

        output = await tool.run(BlastSearchInput(sequence="ACGT" * 30, gap_length=10))

        assert output.succeeded
        assert output.total_hits == 1

    async def test_fetches_are_capped_so_a_wide_search_cannot_flood_ncbi(self) -> None:
        hits = [{**BRACKETING_HIT, "hit_acc": f"REF_{index}"} for index in range(40)]
        ncbi = FakeNCBIClient()
        tool = make_tool(payload_with(*hits), ncbi)

        output = await tool.run(BlastSearchInput(sequence="ACGT" * 30, gap_length=10))

        assert output.total_hits == 40
        assert len(ncbi.requests) == 12, "one round trip per hit would cost dozens of calls"
