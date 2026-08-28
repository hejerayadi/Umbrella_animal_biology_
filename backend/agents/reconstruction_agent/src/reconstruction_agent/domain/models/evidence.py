"""What a tool learned, in a shape that merges.

A returned sequence that cannot say where it came from is indistinguishable
from a fabricated one. That is the whole argument for this module: evidence is
accumulated as it is measured, by the tool that measured it, rather than
reconstructed afterwards from whatever happened to survive in state.

Reconstruction-afterwards is what the agent did before, and it failed in a way
worth recording: the graph produced an exact 45-base reconstruction and
reported it with no provenance and zero gap-spanning hits, because the code
that assembled the audit trail lived on a different execution path from the one
that gathered the evidence. Making each tool contribute its own fragment
removes the possibility - a tool that measures something and reports nothing is
now visible as an empty contribution rather than as a silent gap.

Contributions merge associatively so the order tools run in cannot change the
bundle they produce.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from reconstruction_agent.domain.models.result import (
    AlignmentEvidence,
    Evo2Evidence,
    GapEvidence,
    HomologyEvidence,
    Provenance,
)


class EvidenceContribution(BaseModel):
    """One tool's account of what it measured.

    Every field is optional. A tool that measured nothing in a category leaves
    it unset rather than writing a zero, because an unset field merges cleanly
    while a zero overwrites a real measurement made by an earlier tool.
    """

    model_config = ConfigDict(frozen=True)

    #: External services this call actually reached. Recorded from what ran,
    #: never from a fixed list: a claim about which services backed a
    #: reconstruction has to be true of that reconstruction.
    providers: tuple[str, ...] = ()
    #: Reference collections searched, as opaque provider identifiers.
    databases_searched: tuple[str, ...] = ()
    #: Why those collections were chosen, in plain language.
    database_rationale: str = ""

    homology: HomologyEvidence | None = None
    alignment: AlignmentEvidence | None = None
    evo2: Evo2Evidence | None = None
    validation_checks: tuple[str, ...] = ()
    validation_failures: tuple[str, ...] = ()

    @property
    def is_empty(self) -> bool:
        """True when this call had nothing to report."""
        return not (
            self.providers
            or self.databases_searched
            or self.database_rationale
            or self.homology
            or self.alignment
            or self.evo2
            or self.validation_checks
            or self.validation_failures
        )


class EvidenceBundle(BaseModel):
    """Everything measured for one gap, merged contribution by contribution."""

    model_config = ConfigDict(frozen=True)

    providers: tuple[str, ...] = ()
    databases_searched: tuple[str, ...] = ()
    database_rationale: str = ""
    homology: HomologyEvidence | None = None
    alignment: AlignmentEvidence | None = None
    evo2: Evo2Evidence | None = None
    validation_checks: tuple[str, ...] = ()
    validation_failures: tuple[str, ...] = ()

    def merged_with(self, contribution: EvidenceContribution) -> EvidenceBundle:
        """This bundle plus one more contribution.

        Later measurements of the same category replace earlier ones - a replan
        that re-searched has measured the same thing again, and the second
        measurement is the one that describes the evidence actually used.
        Provider and collection lists accumulate instead, because a run that
        consulted a service does not stop having consulted it.
        """
        return EvidenceBundle(
            providers=_union(self.providers, contribution.providers),
            databases_searched=_union(self.databases_searched, contribution.databases_searched),
            database_rationale=contribution.database_rationale or self.database_rationale,
            homology=contribution.homology or self.homology,
            alignment=contribution.alignment or self.alignment,
            evo2=contribution.evo2 or self.evo2,
            validation_checks=_union(self.validation_checks, contribution.validation_checks),
            validation_failures=_union(self.validation_failures, contribution.validation_failures),
        )

    def to_gap_evidence(self) -> GapEvidence:
        """The evidence as it is reported on a finished gap."""
        return GapEvidence(
            homology=self.homology or HomologyEvidence(),
            alignment=self.alignment or AlignmentEvidence(),
            evo2=self.evo2 or Evo2Evidence(),
            validation_checks=self.validation_checks,
            validation_failures=self.validation_failures,
        )

    def to_provenance(self, *, tool_calls: int = 0) -> Provenance:
        """The audit trail: which services ran, what was searched, and why."""
        return Provenance(
            providers=self.providers,
            databases_searched=self.databases_searched,
            database_rationale=self.database_rationale,
            tool_calls=tool_calls,
        )


def _union(left: tuple[str, ...], right: tuple[str, ...]) -> tuple[str, ...]:
    """Both, in first-seen order, without duplicates."""
    seen = list(left)
    seen.extend(item for item in right if item not in seen)
    return tuple(seen)


def merge_evidence(left: EvidenceBundle, right: EvidenceBundle) -> EvidenceBundle:
    """The state reducer for the evidence channel.

    Declared as a reducer rather than done by assignment so that a tool result
    merged into state can never overwrite what an earlier tool measured - the
    exact failure this module exists to prevent.
    """
    return left.merged_with(
        EvidenceContribution(
            providers=right.providers,
            databases_searched=right.databases_searched,
            database_rationale=right.database_rationale,
            homology=right.homology,
            alignment=right.alignment,
            evo2=right.evo2,
            validation_checks=right.validation_checks,
            validation_failures=right.validation_failures,
        )
    )
