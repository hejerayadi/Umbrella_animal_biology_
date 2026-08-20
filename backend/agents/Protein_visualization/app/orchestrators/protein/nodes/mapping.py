"""SIFTS residue mapping.

A UniProt position is never assumed to equal a PDB residue number. When SIFTS
cannot map the requested position the residue stays unhighlighted and a warning is
raised for the critic to weigh.
"""

import re
from dataclasses import replace
from datetime import UTC, datetime
from typing import Any

from backend.agents.Protein_visualization.app.capabilities.residue_mapping import ResidueMappingCapability
from backend.agents.Protein_visualization.app.domain.exceptions import ProteinAgentError
from backend.agents.Protein_visualization.app.domain.models import EvidenceRef, ProteinStructureRequest
from backend.agents.Protein_visualization.app.observability.logging import log_stage
from backend.agents.Protein_visualization.app.orchestrators.protein.nodes._common import (
    degraded,
    executed,
    failure_code,
    logger,
)
from backend.agents.Protein_visualization.app.orchestrators.protein.nodes.names import MAP_RESIDUES
from backend.agents.Protein_visualization.app.orchestrators.protein.state import ProteinWorkflowState

MUTATION_POSITION = re.compile(r"^[A-Z](\d+)[A-Z]$", re.IGNORECASE)


def requested_position(task: ProteinStructureRequest) -> int | None:
    """The residue to map: the explicit position, or the one inside a mutation."""
    if task.residue_position is not None:
        return task.residue_position
    if task.mutation:
        match = MUTATION_POSITION.match(task.mutation.strip())
        if match:
            return int(match.group(1))
    return None


def mapping_required(task: ProteinStructureRequest) -> bool:
    return requested_position(task) is not None or bool(task.requested_regions)


class ResidueMappingNode:
    def __init__(self, capability: ResidueMappingCapability) -> None:
        self.capability = capability

    async def __call__(self, state: ProteinWorkflowState) -> dict[str, Any]:
        task = state["task"]
        protein = state["resolved_protein"]
        structure = state["selected_structure"]
        assert protein is not None

        position = requested_position(task)
        if structure is None or position is None:
            return executed(MAP_RESIDUES, residue_mappings=[])
        if structure.structure_type != "EXPERIMENTAL":
            return executed(
                MAP_RESIDUES,
                residue_mappings=[],
                warnings=[
                    "SIFTS_NOT_APPLICABLE: the selected model is predicted, so no SIFTS mapping "
                    f"exists for UniProt position {position}."
                ],
            )

        try:
            with log_stage(
                logger,
                f"protein.node.{MAP_RESIDUES}",
                node=MAP_RESIDUES,
                capability="residue_mapping",
            ) as outcome:
                mappings = await self.capability.map(
                    replace(task, residue_position=position), protein, structure
                )
                outcome["mappings"] = len(mappings)
        except ProteinAgentError as exc:
            return degraded(
                MAP_RESIDUES,
                failure_code(exc, "SIFTS_UNAVAILABLE", "SIFTS_TIMEOUT"),
                exc,
                residue_mappings=[],
            )

        if not any(mapping.is_observed for mapping in mappings):
            return executed(
                MAP_RESIDUES,
                residue_mappings=mappings,
                warnings=[
                    f"RESIDUE_NOT_OBSERVED: UniProt position {position} is not observed in "
                    f"{structure.external_id}; it is not highlighted."
                ],
            )

        return executed(
            MAP_RESIDUES,
            residue_mappings=mappings,
            evidence=[
                EvidenceRef(
                    provider="PDBe SIFTS",
                    external_id=f"{structure.external_id}:{position}",
                    retrieved_at=datetime.now(UTC).isoformat(),
                    source_url=(
                        f"https://www.ebi.ac.uk/pdbe/api/mappings/uniprot/{structure.external_id.lower()}"
                    ),
                )
            ],
        )
