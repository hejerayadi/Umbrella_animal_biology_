"""A technical failure must never be reported as a scientific finding.

This is the bug class that has appeared three times in this agent, each time in
a different stage, and each time it looked like a result rather than a fault:

- every search timed out, reported as "no homologues found";
- every homologue sequence failed to download, reported as "no homologue spans
  the gap" - while the search had just measured fifty that do;
- and before that, a search against a collection that excluded the target
  clade, reported as an absence of homology rather than as the wrong question.

All three assert something about biology that the run never established. A
caller acting on "no homologue spans this region" will stop looking; a caller
told the download failed will retry. The distinction is the whole difference
between an agent that abstains honestly and one that misleads.
"""

from __future__ import annotations

import pytest
from tests.fakes import POLAR_BEAR_LINEAGE, isolated_settings

from reconstruction_agent.config.settings import Settings
from reconstruction_agent.domain.enums import GapStatus, UnresolvedReason
from reconstruction_agent.domain.models.homology import HomologHit, HomologySearchOutcome
from reconstruction_agent.domain.models.sequence import Gap, GapContext
from reconstruction_agent.domain.models.taxonomy import TargetProfile
from reconstruction_agent.orchestration.budget import BudgetLedger
from reconstruction_agent.orchestration.deadline import Deadline
from reconstruction_agent.services.homology.round import HomologyRound
from reconstruction_agent.services.reconstruction.reconstruction_service import (
    ReconstructionService,
)

pytestmark = pytest.mark.asyncio


def _hit(accession: str) -> HomologHit:
    return HomologHit(
        accession=accession,
        organism="Ursus maritimus",
        identity=0.96,
        query_coverage=1.0,
        query_start=0,
        query_end=1000,
    )


def _context() -> GapContext:
    return GapContext(
        gap=Gap(gap_id="gap_1", start=5000, end=5045),
        source_accession="NC_003428.1",
        left_flank="ACGT" * 125,
        right_flank="TGCA" * 125,
    )


def _profile() -> TargetProfile:
    return TargetProfile(
        scientific_name="Ursus maritimus", tax_id=29073, taxonomy_lineage=POLAR_BEAR_LINEAGE
    )


class _Homology:
    """A search that produced whatever the test says it produced."""

    def __init__(self, round_: HomologyRound) -> None:
        self._round = round_

    async def search_for_gap(
        self, profile: object, context: object, **kwargs: object
    ) -> HomologyRound:
        return self._round


class _Sequences:
    """A homologue download that retrieves whatever the test allows."""

    def __init__(self, retrieved: tuple[HomologHit, ...]) -> None:
        self._retrieved = retrieved

    async def fetch_homolog_sequences(
        self, hits: tuple[HomologHit, ...], **kwargs: object
    ) -> tuple[HomologHit, ...]:
        return self._retrieved


def _service(round_: HomologyRound, retrieved: tuple[HomologHit, ...]) -> ReconstructionService:
    settings: Settings = isolated_settings()
    return ReconstructionService(
        settings=settings,
        sequences=_Sequences(retrieved),  # type: ignore[arg-type]
        taxonomy=None,  # type: ignore[arg-type]
        homology=_Homology(round_),  # type: ignore[arg-type]
        mafft=None,  # type: ignore[arg-type]
        candidates=None,  # type: ignore[arg-type]
        engine=None,  # type: ignore[arg-type]
    )


async def _run(round_: HomologyRound, retrieved: tuple[HomologHit, ...]) -> object:
    service = _service(round_, retrieved)
    return await service._reconstruct_gap(
        _context(),
        _profile(),
        deadline=Deadline(total_seconds=300.0, reserve_seconds=30.0),
        budget=BudgetLedger(),
    )


async def test_a_search_that_never_completed_is_not_an_absence_of_homologues() -> None:
    """Every search timed out. Nothing was learned about the biology."""
    result = await _run(
        HomologyRound(
            gap_id="gap_1",
            outcomes=(
                HomologySearchOutcome(
                    database_code="core_nt@txid9632", error="search did not finish in time"
                ),
            ),
        ),
        retrieved=(),
    )

    assert result.status is GapStatus.UNRESOLVED  # type: ignore[attr-defined]
    assert result.unresolved_reason is UnresolvedReason.DEADLINE_EXCEEDED  # type: ignore[attr-defined]


async def test_a_completed_search_with_nothing_crossing_is_an_absence_of_homologues() -> None:
    """The search worked and found nothing usable. That IS a finding."""
    result = await _run(
        HomologyRound(
            gap_id="gap_1",
            outcomes=(
                HomologySearchOutcome(
                    database_code="core_nt@txid9632", hits=(_hit("X1"),), gap_spanning_hits=0
                ),
            ),
        ),
        retrieved=(),
    )

    assert result.unresolved_reason is UnresolvedReason.NO_HOMOLOGS_FOUND  # type: ignore[attr-defined]


async def test_failing_to_download_the_homologues_is_not_an_absence_of_them() -> None:
    """Fifty homologues crossed the gap and none could be fetched.

    Reporting INSUFFICIENT_GAP_SPANNING_HOMOLOGS here would deny evidence the
    run had already measured, and would tell a caller to stop looking for
    something that exists.
    """
    spanning = tuple(_hit(f"ACC{i}") for i in range(50))
    result = await _run(
        HomologyRound(
            gap_id="gap_1",
            outcomes=(
                HomologySearchOutcome(
                    database_code="core_nt@txid9632", hits=spanning, gap_spanning_hits=50
                ),
            ),
        ),
        retrieved=(),
    )

    assert result.unresolved_reason is UnresolvedReason.EVIDENCE_RETRIEVAL_FAILED  # type: ignore[attr-defined]
    assert "none of their sequences could be retrieved" in result.explanation  # type: ignore[attr-defined]
    assert result.selected_candidate is None  # type: ignore[attr-defined]


async def test_the_reported_evidence_still_records_what_the_search_found() -> None:
    """An abstention must not erase the measurement that led to it."""
    spanning = tuple(_hit(f"ACC{i}") for i in range(50))
    result = await _run(
        HomologyRound(
            gap_id="gap_1",
            outcomes=(
                HomologySearchOutcome(
                    database_code="core_nt@txid9632", hits=spanning, gap_spanning_hits=50
                ),
            ),
        ),
        retrieved=(),
    )

    assert result.evidence.homology.gap_spanning_hits == 50  # type: ignore[attr-defined]
