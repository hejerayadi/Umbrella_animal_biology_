"""Resolve the target's biology: organism, tax id, lineage, molecule type.

Separate from fetching the sequence because it answers a different question and
fails independently. A record whose organism cannot be resolved is still
reconstructible - the search simply loses its taxonomic scoping and falls back
to a wider one - so a failure here degrades the plan rather than ending it.

What comes back is a `TargetProfile`, which carries biology only. No provider
vocabulary appears on it: the profile must stay true if the homology provider
is replaced tomorrow.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict

from reconstruction_agent.domain.enums import ToolName
from reconstruction_agent.domain.exceptions import ReconstructionError
from reconstruction_agent.domain.models.evidence import EvidenceContribution
from reconstruction_agent.domain.models.sequence import SequenceRecord
from reconstruction_agent.domain.models.taxonomy import TargetProfile
from reconstruction_agent.services.taxonomy.taxonomy_service import TaxonomyService
from reconstruction_agent.tools.base import Tool, ToolOutcome


class AssemblyMetadataInput(BaseModel):
    model_config = ConfigDict(frozen=True)

    record: SequenceRecord
    #: The caller's own name for the organism, preferred over the record's when
    #: present: the request is more specific than a database annotation.
    scientific_name: str | None = None


class AssemblyMetadataOutput(BaseModel):
    model_config = ConfigDict(frozen=True)

    profile: TargetProfile


class GetAssemblyMetadataTool(Tool[AssemblyMetadataInput, AssemblyMetadataOutput]):
    """Organism, tax id, lineage and molecule type for the target."""

    name = ToolName.GET_ASSEMBLY_METADATA
    description = (
        "Resolve the target organism against NCBI Taxonomy, giving the lineage that "
        "scopes the homology search and the distances that weight its hits."
    )
    input_model = AssemblyMetadataInput

    def __init__(self, taxonomy: TaxonomyService) -> None:
        self._taxonomy = taxonomy

    async def run(self, request: AssemblyMetadataInput) -> ToolOutcome[AssemblyMetadataOutput]:
        record = request.record
        name = request.scientific_name or record.organism or "unknown organism"
        try:
            profile = await self._taxonomy.profile_for(name, record.tax_id, record.molecule_type)
        except ReconstructionError as error:
            return ToolOutcome(tool=self.name, ok=False, reason=str(error), transport_error=True)

        if profile.tax_id is None:
            # Degraded, not failed: the search can still run, just unscoped.
            return ToolOutcome(
                tool=self.name,
                ok=True,
                data=AssemblyMetadataOutput(profile=profile),
                reason=f"{name} did not resolve to a taxon; the search will not be scoped.",
            )

        return ToolOutcome(tool=self.name, ok=True, data=AssemblyMetadataOutput(profile=profile))

    def summarise(self, outcome: ToolOutcome[AssemblyMetadataOutput]) -> dict[str, Any]:
        if outcome.data is None:
            return {}
        profile = outcome.data.profile
        return {
            "organism": profile.scientific_name,
            "tax_id": profile.tax_id,
            "lineage_depth": len(profile.taxonomy_lineage),
            "molecule_type": profile.molecule_type.value,
        }

    def contribute(self, outcome: ToolOutcome[AssemblyMetadataOutput]) -> EvidenceContribution:
        if outcome.data is None:
            return EvidenceContribution()
        return EvidenceContribution(providers=("NCBI Taxonomy",))
