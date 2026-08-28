"""A slow scope must not be allowed to finish the run.

Measured live on NC_003428.1: the narrowest scope was still queued at 270 s
while two wider ones had returned 50 gap-spanning hits each at 53 s and 72 s.
The round waited for the front runner, the working window closed, and the run
ended UNRESOLVED holding evidence it never had time to fetch. These tests pin
the bound that prevents that.
"""

from __future__ import annotations

import asyncio

from reconstruction_agent.domain.models.homology import HomologySearchOutcome
from reconstruction_agent.services.homology.homology_service import HomologyService


def _outcome(code: str, *, spanning: int) -> HomologySearchOutcome:
    return HomologySearchOutcome(
        database_code=code,
        database_label=code,
        total_hits=spanning,
        gap_spanning_hits=spanning,
        duration_seconds=0.0,
    )


async def _after(delay: float, outcome: HomologySearchOutcome) -> HomologySearchOutcome:
    await asyncio.sleep(delay)
    return outcome


class TestTheNarrowestScopeIsPreferredButBounded:
    async def test_the_front_runner_is_waited_for_when_it_is_merely_slower(self) -> None:
        """Arriving first is not evidence of being right: the narrowest scope
        gives the closest relatives and is worth a short wait."""
        collected = await HomologyService._race(
            [
                _after(0.05, _outcome("narrow", spanning=10)),
                _after(0.0, _outcome("wide", spanning=50)),
            ],
            grace_seconds=5.0,
        )

        assert {o.database_code for o in collected} == {"narrow", "wide"}

    async def test_a_stalled_front_runner_does_not_hold_the_round(self) -> None:
        """The measured failure: the round must return the evidence it has
        rather than waiting out a queue that may never drain."""
        started = asyncio.get_running_loop().time()
        collected = await HomologyService._race(
            [
                _after(30.0, _outcome("narrow", spanning=10)),
                _after(0.0, _outcome("wide", spanning=50)),
            ],
            grace_seconds=0.1,
        )
        elapsed = asyncio.get_running_loop().time() - started

        assert [o.database_code for o in collected] == ["wide"]
        assert elapsed < 1.0, "the round waited out the stalled scope"

    async def test_the_grace_only_starts_once_evidence_is_usable(self) -> None:
        """A scope returning nothing usable is not a reason to stop waiting -
        otherwise an empty fast result would cancel the search that works."""
        collected = await HomologyService._race(
            [
                _after(0.15, _outcome("narrow", spanning=10)),
                _after(0.0, _outcome("empty", spanning=0)),
            ],
            grace_seconds=0.05,
        )

        assert {o.database_code for o in collected} == {"narrow", "empty"}
