"""The BLAST databases EBI really offers, and what they actually mean.

Two jobs, both of which exist because of measured failures.

**Validation.** `_RELAXED_DATABASE` used to be `em_rel_vrt`, a code nobody
checked and which does not exist. EBI validates the database name against a
fixed list and rejects anything else with a bare HTTP 400, so *every* relaxed
retry died on submission and the run silently lost half its search budget.
Composing a code from a division makes that class of typo easy to reintroduce,
so nothing reaches the wire without being checked against this catalogue first.

**Explanation.** The planner picks the databases, and a bare list of 284 codes
invites it to make exactly the mistake the codebase already made: nothing in the
string `em_std_vrt` tells you it *excludes mammals*. So the shortlist handed to
the planner carries the semantics with it - which divisions are mutually
exclusive, which subdivision holds draft scaffolds, and which databases were
measured too slow to finish inside an orchestrator slice.

The live catalogue is always preferred. `ena_snapshot.SNAPSHOT` stands in when
EBI is unreachable, so an offline run still validates against something real.
"""
from __future__ import annotations

from dataclasses import dataclass

from configuration.logging import get_logger
from domain.exceptions import ReconstructionError
from domain.services.target_profile import Molecule, TargetProfile, candidate_databases
from infrastructure.embl_ebi.blast_client import BlastClient
from tools.blast.ena_snapshot import SNAPSHOT

_log = get_logger(__name__)

#: The semantics the planner cannot infer from the codes themselves. This text
#: is the whole reason the discovery step is a tool and not a constant.
DIVISION_SEMANTICS = (
    "EMBL/ENA taxonomic divisions are MUTUALLY EXCLUSIVE, not nested. "
    "'vrt' (Vertebrate) means OTHER vertebrates: it EXCLUDES mammals, human, "
    "mouse and rodent, which live in 'mam', 'hum', 'mus' and 'rod' respectively. "
    "Searching 'vrt' for a mammal returns fish and birds and cannot find the "
    "right homologues at any e-value. Choose the division the target organism "
    "itself is filed under. "
    "Within a division, 'em_std_<div>' holds finished records (a published "
    "mitogenome), while 'em_<div>' is the whole division and also carries the "
    "WGS, CON and HTG entries where draft genomic scaffolds live."
)

#: Measured against the real service on 2026-08-25 with a 1 kb query: `em_all`
#: took 749 s and `em_std` had not finished at 813 s, against a 600 s
#: orchestrator slice. They return correct answers and are still unusable.
TOO_SLOW: dict[str, str] = {
    "em_std": "measured >813 s, never finished - exceeds the 600 s slice",
    "em_all": "measured 749 s - exceeds the 600 s slice",
}


@dataclass(frozen=True, slots=True)
class DatabaseOption:
    """One database the planner may choose, with the reason it might want to."""

    code: str
    label: str
    #: True when the target's taxonomy prior points at this database.
    likely_for_target: bool = False
    #: Set when the database is valid but known to blow the slice budget.
    too_slow: str | None = None

    def as_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "code": self.code,
            "label": self.label,
            "likely_for_target": self.likely_for_target,
        }
        if self.too_slow:
            payload["too_slow"] = self.too_slow
        return payload


class EbiDatabaseCatalogue:
    """The nucleotide database codes EBI accepts, fetched once and cached.

    Cached for the life of the process: the list changes on EBI's release
    schedule, not within a run, and a 100 kB fetch per BLAST call would be pure
    waste on an agent that already spends minutes per search.
    """

    def __init__(self, client: BlastClient | None = None) -> None:
        self._client = client
        self._codes: dict[str, str] | None = None

    async def codes(self) -> dict[str, str]:
        """Every valid nucleotide database code, mapped to its label.

        Never raises. A catalogue that cannot be fetched falls back to the
        snapshot, because failing to *describe* the databases must not stop the
        agent from *searching* them.
        """
        if self._codes is not None:
            return self._codes

        if self._client is not None:
            try:
                live = await self._client.database_codes()
            except ReconstructionError as error:
                _log.info("ena_catalogue_unavailable", error=str(error))
            else:
                if live:
                    self._codes = live
                    _log.info("ena_catalogue_loaded", source="live", count=len(live))
                    return self._codes

        self._codes = dict(SNAPSHOT)
        _log.info("ena_catalogue_loaded", source="snapshot", count=len(self._codes))
        return self._codes

    async def validate(self, codes: list[str]) -> tuple[list[str], list[str]]:
        """Split proposed codes into (accepted, rejected).

        Order is preserved and duplicates are dropped: the planner may name the
        same database twice, and submitting it twice would spend two of eight
        BLAST calls proving the same point.
        """
        known = await self.codes()
        accepted: list[str] = []
        rejected: list[str] = []
        for code in codes:
            name = (code or "").strip()
            if not name or name in accepted or name in rejected:
                continue
            (accepted if name in known else rejected).append(name)

        if rejected:
            _log.warning("blast_database_rejected", codes=rejected, detail="not in EBI catalogue")
        return accepted, rejected

    async def shortlist(
        self, profile: TargetProfile, *, limit: int = 12
    ) -> list[DatabaseOption]:
        """The databases worth showing the planner for this target.

        Not the whole catalogue: 284 nucleotide codes would crowd out the rest of
        the prompt and most of them - EST, GSS, STS, patent divisions - cannot
        carry a genomic gap. What is offered is the target's own division family
        first, then the other divisions so the planner can overrule the prior,
        which is the point of asking it at all.
        """
        known = await self.codes()
        prior = set(await self._valid_prior(profile))

        options: list[DatabaseOption] = []
        seen: set[str] = set()

        def offer(code: str) -> None:
            if code in seen or code not in known:
                return
            seen.add(code)
            options.append(
                DatabaseOption(
                    code=code,
                    label=known[code],
                    likely_for_target=code in prior,
                    too_slow=TOO_SLOW.get(code),
                )
            )

        for code in await self._valid_prior(profile):
            offer(code)

        # The same two shapes for every division, so the planner can see what
        # choosing a different division would actually mean.
        for division in ("hum", "mus", "rod", "mam", "vrt", "inv", "pln", "fun", "pro"):
            offer(f"em_std_{division}")
            offer(f"em_{division}")
            if len(options) >= limit:
                break

        return options[:limit]

    async def _valid_prior(self, profile: TargetProfile) -> list[str]:
        accepted, _ = await self.validate(list(candidate_databases(profile)))
        return accepted

    async def resolve(
        self,
        profile: TargetProfile,
        proposed: list[str] | None = None,
        *,
        limit: int = 3,
    ) -> list[str]:
        """The databases to actually search, validated, prior-seeded, never empty-by-surprise.

        `proposed` is what the planner asked for. Anything invalid is dropped
        rather than submitted; if that leaves nothing, the taxonomy prior takes
        over, which is what keeps an offline or unhelpful planner from stalling
        the run.

        Returns `[]` only when the prior is empty too - the genuinely unknown
        organism - and the caller then probes rather than guessing.
        """
        accepted: list[str] = []
        if proposed:
            accepted, _ = await self.validate(proposed)

        # Never search a database measured as unable to finish in a slice.
        usable = [code for code in accepted if code not in TOO_SLOW]
        if len(usable) < len(accepted):
            _log.info(
                "blast_database_dropped_too_slow",
                dropped=[code for code in accepted if code in TOO_SLOW],
            )

        prior = await self._valid_prior(profile)
        if not usable:
            return prior[:limit]

        # The prior *augments* the planner's choice rather than losing to it.
        # If the model picks `em_std_vrt` for a mammal - the exact mistake this
        # whole change exists to correct - searching only that repeats the
        # failure, while searching only the prior would make asking the planner
        # pointless. Probing both costs nothing in wall clock, because the round
        # is dispatched with `asyncio.gather`, and the carrier count then
        # settles it on evidence instead of on either guess.
        merged = list(usable)
        for code in prior:
            if len(merged) >= limit:
                break
            if code not in merged:
                merged.append(code)
        return merged


def molecule_hint(profile: TargetProfile) -> str:
    """One line telling the planner which subdivision shape fits this target."""
    if profile.molecule is Molecule.NUCLEAR:
        return (
            "The target is a nuclear/draft assembly, so prefer 'em_<div>' - only "
            "the whole division carries WGS/CON/HTG draft scaffolds."
        )
    if profile.molecule is Molecule.ORGANELLAR:
        return (
            "The target is an organellar genome, so prefer 'em_std_<div>' - "
            "finished records, a smaller and faster search."
        )
    return (
        "The molecule type is unknown; 'em_std_<div>' is the cheaper first "
        "search, with 'em_<div>' as the widening step."
    )
