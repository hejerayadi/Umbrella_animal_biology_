"""Picking the canonical UniProt entry out of a real gene search.

The shapes here are the ones rest.uniprot.org actually returns: a gene symbol
search answers with the reviewed Swiss-Prot entry alongside several unreviewed
TrEMBL records for the same protein.
"""

from typing import Any

import pytest

from backend.agents.Protein_visualization.app.capabilities.identity import IdentityCapability
from backend.agents.Protein_visualization.app.domain.exceptions import ProteinNotFoundError
from backend.agents.Protein_visualization.tests.factories import agent_task
from backend.agents.Protein_visualization.tests.fakes import FakeUniProt, fixture

REVIEWED = "UniProtKB reviewed (Swiss-Prot)"
UNREVIEWED = "UniProtKB unreviewed (TrEMBL)"


def _entry(accession: str, entry_type: str) -> dict[str, Any]:
    return {**fixture("uniprot_P04637.json"), "primaryAccession": accession, "entryType": entry_type}


async def _resolve(records: list[dict[str, Any]]) -> Any:
    capability = IdentityCapability(FakeUniProt(matches=records))  # type: ignore[arg-type]
    protein, _ = await capability.resolve(agent_task(uniprot_accession=None).to_domain())
    return protein


async def test_the_reviewed_entry_wins_over_trembl_fragments() -> None:
    """What `gene_exact:TP53 AND organism_name:"Homo sapiens"` really returns."""
    protein = await _resolve(
        [
            _entry("P04637", REVIEWED),
            _entry("K7PPA8", UNREVIEWED),
            _entry("E9PFT5", UNREVIEWED),
            _entry("S4R334", UNREVIEWED),
            _entry("H2EHT1", UNREVIEWED),
        ]
    )

    assert protein.uniprot_accession == "P04637"


async def test_a_lone_unreviewed_entry_is_still_usable() -> None:
    """Most species have no curated entry at all; one candidate is not ambiguous."""
    protein = await _resolve([_entry("K7PPA8", UNREVIEWED)])

    assert protein.uniprot_accession == "K7PPA8"


async def test_several_unreviewed_candidates_stay_ambiguous() -> None:
    with pytest.raises(ProteinNotFoundError, match="ambiguous"):
        await _resolve([_entry("K7PPA8", UNREVIEWED), _entry("E9PFT5", UNREVIEWED)])


async def test_two_reviewed_entries_stay_ambiguous() -> None:
    """Curation cannot arbitrate between two of its own entries."""
    with pytest.raises(ProteinNotFoundError, match="ambiguous"):
        await _resolve([_entry("P04637", REVIEWED), _entry("P04638", REVIEWED)])


async def test_no_candidate_at_all_is_reported_as_not_found() -> None:
    with pytest.raises(ProteinNotFoundError, match="no protein"):
        await _resolve([])
