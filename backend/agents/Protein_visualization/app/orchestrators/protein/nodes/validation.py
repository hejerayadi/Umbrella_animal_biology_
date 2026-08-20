"""Contract validation: species, identity anchor, and residue coherence."""

import re
from typing import Any

from backend.agents.Protein_visualization.app.domain.enums import AnalysisStatus
from backend.agents.Protein_visualization.app.observability.logging import log_stage
from backend.agents.Protein_visualization.app.orchestrators.protein.nodes._common import executed, logger
from backend.agents.Protein_visualization.app.orchestrators.protein.nodes.names import VALIDATE_INPUT
from backend.agents.Protein_visualization.app.orchestrators.protein.state import ProteinWorkflowState

ACCESSION_PATTERN = re.compile(r"^[OPQ][0-9][A-Z0-9]{3}[0-9]$|^[A-NR-Z][0-9]([A-Z][A-Z0-9]{2}[0-9]){1,2}$")
SEQUENCE_PATTERN = re.compile(r"^[ACDEFGHIKLMNPQRSTVWYBXZUO]+$", re.IGNORECASE)
MUTATION_PATTERN = re.compile(r"^[A-Z]\d+[A-Z]$", re.IGNORECASE)


async def validate_input(state: ProteinWorkflowState) -> dict[str, Any]:
    task = state["task"]
    problems: list[str] = []

    with log_stage(
        logger,
        f"protein.node.{VALIDATE_INPUT}",
        node=VALIDATE_INPUT,
        capability="input_validation",
    ) as outcome:
        if task.uniprot_accession and not ACCESSION_PATTERN.match(task.uniprot_accession.upper()):
            problems.append(f"'{task.uniprot_accession}' is not a valid UniProt accession.")
        if task.protein_sequence and not SEQUENCE_PATTERN.match(task.protein_sequence.strip()):
            problems.append("protein_sequence contains characters outside the amino-acid alphabet.")
        if not task.uniprot_accession and not task.protein_sequence and not task.resolved_gene_id:
            problems.append("One of uniprot_accession, protein_sequence, or resolved_gene_id is required.")
        if task.mutation and not MUTATION_PATTERN.match(task.mutation.strip()):
            problems.append(f"'{task.mutation}' is not a recognised point-mutation notation (e.g. R273H).")
        if task.species.taxon_id <= 0:
            problems.append("species.taxon_id must be a positive NCBI taxonomy identifier.")
        outcome["problems"] = len(problems)

    if problems:
        return executed(
            VALIDATE_INPUT,
            current_status=AnalysisStatus.needs_clarification,
            clarification=" ".join(problems),
        )
    return executed(VALIDATE_INPUT, current_status=AnalysisStatus.validating)
