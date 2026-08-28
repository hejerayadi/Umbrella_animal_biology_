"""Scoring the agent against bases it was not shown, through the real services.

Every other test in this suite mocks the network edge, which proves the code
runs but cannot tell a working evidence path from a plausible-looking one. Only
this can: a known region is withheld, the agent reconstructs it from homology
alone, and the answer is compared against what was actually there.

It is the test that would have caught the failure this design exists to
prevent. Searching a polar bear against a collection that excludes mammals
returns hits, produces an alignment and yields a candidate - everything looks
like it worked, and the sequence is wrong.

Marked `external` and skipped unless RUN_EXTERNAL_TESTS=1. It costs several
minutes of EMBL-EBI queue time, and it is unkind to run it on every commit.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from reconstruction_agent.config.settings import get_settings
from reconstruction_agent.integrations.ncbi.client import NcbiClient
from reconstruction_agent.main import create_app

pytestmark = [pytest.mark.external, pytest.mark.asyncio]

#: Polar bear mitogenome. Chosen because it is small, well covered by close
#: relatives, and the exact case the database-selection design was built from.
ACCESSION = "NC_003428.1"
GAP_START = 5000
GAP_LENGTH = 45


@pytest.fixture
async def withheld_bases() -> str:
    """The bases the agent will not be shown."""
    ncbi = NcbiClient(get_settings().ncbi)
    try:
        record = await ncbi.fetch_sequence(ACCESSION)
        return record.residues[GAP_START : GAP_START + GAP_LENGTH]
    finally:
        await ncbi.aclose()


async def test_a_withheld_region_is_recovered_exactly(withheld_bases: str) -> None:
    """The end-to-end claim, measured rather than asserted by construction.

    Expected behaviour, all of which has been observed live: the target
    resolves to *Ursus maritimus*; the selector ranks the mammal-covering
    collection first from taxonomy alone and records why; around fifty
    homologues cross the gap; and all 45 bases come back exactly, at high
    confidence.
    """
    client = TestClient(create_app(), raise_server_exceptions=False)

    response = client.post(
        "/execute",
        json={
            "instruction": "Reconstruct the unresolved region.",
            "context": {
                "sequence_accession": ACCESSION,
                "target_gaps": [{"start": GAP_START, "end": GAP_START + GAP_LENGTH}],
            },
        },
        headers={"X-Trace-Id": "ground-truth"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "completed"

    gap = body["output"]["reconstruction"]["reconstructions"][0]
    assert gap["status"] == "RESOLVED", gap.get("explanation")

    candidate = gap["selected_candidate"]
    assert candidate["sequence"] == withheld_bases
    assert candidate["confidence_level"] == "HIGH"

    # The rationale must show the choice came from taxonomy, not from a
    # database code written down somewhere.
    assert "covers Mammalia" in gap["provenance"]["database_rationale"]
    assert gap["evidence"]["homology"]["gap_spanning_hits"] > 10
