"""Which databases this run should probe, and the evidence for choosing one.

The agent no longer names a database. It discovers what EBI offers, proposes a
small candidate set, searches them **in parallel**, and then keeps whichever one
actually produced references that cross the gap. This module owns the first half
of that - the proposal - and `record_trial` / `best_database` own the second.

Placed here rather than in `ToolSelector` because the proposal needs the network
(an NCBI lineage lookup and EBI's catalogue) while the selector is synchronous.
The graph resolves it once per run in `plan`, and every gap afterwards reuses the
answer.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

from configuration.logging import get_logger
from domain.exceptions import ReconstructionError
from domain.services.target_profile import (
    Molecule,
    TargetProfile,
    classify_division,
    classify_molecule,
    organism_from_description,
)
from infrastructure.ncbi.taxonomy import TaxonomyService
from tools.blast.catalogue import (
    DIVISION_SEMANTICS,
    EbiDatabaseCatalogue,
    molecule_hint,
)

_log = get_logger(__name__)

#: How many databases one probe round may search at once. Three fits the slice -
#: the round is dispatched with `asyncio.gather`, so the wall clock is the
#: slowest search (~300 s measured), not the sum - while leaving most of the
#: eight-call BLAST budget for the gaps that follow.
MAX_PROBE_DATABASES = 3

#: A lineage lookup is an enrichment, never a precondition. `NCBIClient` retries
#: three times with backoff on transport errors, so an unreachable NCBI could
#: otherwise spend most of a slice failing before BLAST is even submitted.
_TAXONOMY_TIMEOUT = 8.0


@dataclass(frozen=True, slots=True)
class DatabaseAdvice:
    """What to search, why, and what to tell the planner about the alternatives."""

    candidates: list[str]
    profile: TargetProfile
    #: Serialisable options for the planner prompt.
    options: list[dict[str, Any]]
    semantics: str = DIVISION_SEMANTICS
    hint: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "candidates": list(self.candidates),
            "division": self.profile.division.value if self.profile.division else None,
            "molecule": self.profile.molecule.value,
            "source": self.profile.source,
            "options": self.options,
            "semantics": self.semantics,
            "hint": self.hint,
        }


class DatabaseAdvisor:
    """Proposes the databases worth probing for one target."""

    def __init__(
        self,
        catalogue: EbiDatabaseCatalogue,
        taxonomy: TaxonomyService | None = None,
    ) -> None:
        self._catalogue = catalogue
        self._taxonomy = taxonomy
        self._profiles: dict[tuple[str | None, str], TargetProfile] = {}

    async def advise(
        self,
        *,
        organism: str | None,
        description: str | None = None,
        instruction: str | None = None,
        assembly_level: str | None = None,
        length: int | None = None,
        proposed: list[str] | None = None,
    ) -> DatabaseAdvice:
        """The candidate databases for this target, prior-seeded and validated.

        `proposed` is whatever the planner asked for. It is honoured where valid
        and *augmented* by the taxonomy prior rather than replaced by it: if the
        model picks a division the organism is not filed under - the exact
        mistake this change exists to correct - probing both costs nothing extra
        in wall clock and lets the carrier count settle it on evidence.
        """
        profile = await self.profile(
            organism=organism,
            description=description,
            instruction=instruction,
            assembly_level=assembly_level,
            length=length,
        )
        candidates = await self._catalogue.resolve(
            profile, proposed, limit=MAX_PROBE_DATABASES
        )
        options = [option.as_dict() for option in await self._catalogue.shortlist(profile)]

        _log.info(
            "blast_databases_advised",
            organism=organism,
            division=profile.division.value if profile.division else None,
            molecule=profile.molecule.value,
            source=profile.source,
            candidates=candidates,
            proposed=proposed or [],
        )
        return DatabaseAdvice(
            candidates=candidates,
            profile=profile,
            options=options,
            hint=molecule_hint(profile),
        )

    async def profile(
        self,
        *,
        organism: str | None,
        description: str | None = None,
        instruction: str | None = None,
        assembly_level: str | None = None,
        length: int | None = None,
    ) -> TargetProfile:
        """What kind of target this is. Cached per (organism, molecule).

        The record's own description is the authority and is tried alone first.
        Only when it settles nothing does the orchestrator's instruction get a
        say - it is prose written for a person and mentions the target in
        passing ("its largest genomic scaffold"), which is a real signal but a
        weaker one, and letting it outrank the defline would let a request that
        merely name-drops mitochondria misfile a nuclear scaffold.
        """
        molecule = classify_molecule(
            description=description, assembly_level=assembly_level, length=length
        )
        if molecule is Molecule.UNKNOWN and instruction:
            molecule = classify_molecule(description=instruction, length=length)
        # The organism may only be recoverable from the record's own defline -
        # a pasted sequence carries no species at all, but a fetched one does.
        name = organism or organism_from_description(description)
        key = (name, molecule.value)
        cached = self._profiles.get(key)
        if cached is not None:
            return cached

        lineage = await self._lineage(name)
        division = classify_division(lineage)
        profile = TargetProfile(
            division=division,
            molecule=molecule,
            source="lineage" if division else "none",
        )
        self._profiles[key] = profile
        return profile

    async def _lineage(self, organism: str | None) -> tuple[str, ...]:
        """The target's NCBI lineage, or `()` if it cannot be had in time."""
        if not organism or self._taxonomy is None:
            return ()
        try:
            return await asyncio.wait_for(
                self._taxonomy.lineage(organism), timeout=_TAXONOMY_TIMEOUT
            )
        except TimeoutError:
            _log.info("taxonomy_timed_out", organism=organism, seconds=_TAXONOMY_TIMEOUT)
        except ReconstructionError as error:
            _log.info("taxonomy_unavailable", organism=organism, error=str(error))
        return ()


def record_trial(
    trials: dict[str, Any], database: str | None, *, hits: int, carrying_gap: int, seconds: float
) -> dict[str, Any]:
    """One database's result, merged into the running tally.

    Accumulated rather than overwritten: a database probed on one gap and reused
    on the next has learned more about itself each time, and the totals are what
    the comparison is made on.
    """
    if not database:
        return trials
    current = dict(trials.get(database) or {})
    return {
        **trials,
        database: {
            "hits": int(current.get("hits", 0)) + hits,
            "carrying_gap": int(current.get("carrying_gap", 0)) + carrying_gap,
            "seconds": round(float(current.get("seconds", 0.0)) + seconds, 1),
            "searches": int(current.get("searches", 0)) + 1,
        },
    }


def best_database(trials: dict[str, Any]) -> str | None:
    """The database that produced the most gap carriers, or None if none did.

    Carriers first, because that is the only count that predicts whether an
    alignment can produce a fill - a database can return fifty hits and not one
    of them reaching across the gap, which is precisely what `em_std_vrt` did on
    every mammal run. Total hits break a tie, then speed, then the name so the
    choice is deterministic.

    Returns None when nothing carried anything: there is no winner to keep, and
    claiming one would send every later gap to a database already shown not to
    work.
    """
    ranked = [
        (
            int((stats or {}).get("carrying_gap", 0)),
            int((stats or {}).get("hits", 0)),
            -float((stats or {}).get("seconds", 0.0)),
            code,
        )
        for code, stats in (trials or {}).items()
    ]
    winners = [row for row in ranked if row[0] > 0]
    if not winners:
        return None
    winners.sort(key=lambda row: (-row[0], -row[1], -row[2], row[3]))
    return winners[0][3]


def is_nuclear(profile: TargetProfile) -> bool:
    """Whether the target is a nuclear/draft assembly rather than an organelle."""
    return profile.molecule is Molecule.NUCLEAR
