"""Experimental search, deterministic evaluation, AlphaFold fallback, selection."""

from datetime import UTC, datetime
from typing import Any

from backend.agents.Protein_visualization.app.capabilities.structures import StructureCapability
from backend.agents.Protein_visualization.app.domain.enums import AnalysisStatus, PreferredSource
from backend.agents.Protein_visualization.app.domain.exceptions import ProteinAgentError
from backend.agents.Protein_visualization.app.domain.models import EvidenceRef, StructureCandidate
from backend.agents.Protein_visualization.app.observability.logging import log_stage
from backend.agents.Protein_visualization.app.orchestrators.protein.nodes._common import (
    degraded,
    executed,
    failure_code,
    logger,
)
from backend.agents.Protein_visualization.app.orchestrators.protein.nodes.names import (
    EVALUATE_PDB,
    SEARCH_ALPHAFOLD,
    SEARCH_EXPERIMENTAL,
    SELECT_STRUCTURE,
)
from backend.agents.Protein_visualization.app.orchestrators.protein.state import ProteinWorkflowState


def _evidence(provider: str, candidate: StructureCandidate) -> EvidenceRef:
    return EvidenceRef(
        provider=provider,
        external_id=candidate.external_id,
        retrieved_at=datetime.now(UTC).isoformat(),
        source_url=candidate.file_url,
    )


class StructureNodes:
    """The only place a structure is chosen. The LLM is never consulted here."""

    def __init__(self, capability: StructureCapability, min_sequence_coverage: float) -> None:
        self.capability = capability
        self.min_sequence_coverage = min_sequence_coverage

    async def search_experimental(self, state: ProteinWorkflowState) -> dict[str, Any]:
        protein = state["resolved_protein"]
        assert protein is not None
        if state["task"].preferred_source is PreferredSource.alphafold:
            return executed(SEARCH_EXPERIMENTAL, pdb_candidates=[])
        try:
            with log_stage(logger, SEARCH_EXPERIMENTAL, node=SEARCH_EXPERIMENTAL) as outcome:
                candidates = await self.capability.experimental(protein)
                outcome["candidates"] = len(candidates)
        except ProteinAgentError as exc:
            return degraded(
                SEARCH_EXPERIMENTAL,
                failure_code(exc, "PDB_UNAVAILABLE", "PDB_TIMEOUT"),
                exc,
                pdb_candidates=[],
            )
        return executed(
            SEARCH_EXPERIMENTAL,
            pdb_candidates=candidates,
            evidence=[_evidence("RCSB PDB", candidate) for candidate in candidates],
        )

    def evaluate_pdb(self, state: ProteinWorkflowState) -> dict[str, Any]:
        """Keep only candidates whose identity, chain, coverage and metadata hold up."""
        valid: list[StructureCandidate] = []
        rejected: list[str] = []
        with log_stage(logger, EVALUATE_PDB, node=EVALUATE_PDB) as outcome:
            for candidate in state["pdb_candidates"]:
                reason = self._rejection_reason(candidate)
                if reason:
                    rejected.append(f"{candidate.external_id} ({reason})")
                else:
                    valid.append(candidate)
            outcome["valid"] = len(valid)
            outcome["rejected"] = len(rejected)

        warnings = [f"PDB_CANDIDATES_REJECTED: {', '.join(rejected)}"] if rejected and not valid else []
        return executed(EVALUATE_PDB, valid_pdb_candidates=valid, warnings=warnings)

    def _rejection_reason(self, candidate: StructureCandidate) -> str | None:
        if candidate.chain_id is None:
            return "no polymer chain matched the protein"
        if candidate.sequence_coverage < self.min_sequence_coverage:
            return f"sequence coverage {candidate.sequence_coverage:.0%} below the required minimum"
        if candidate.experimental_method is None and candidate.resolution_angstrom is None:
            return "no experimental metadata available"
        return None

    async def search_alphafold(self, state: ProteinWorkflowState) -> dict[str, Any]:
        protein = state["resolved_protein"]
        assert protein is not None
        try:
            with log_stage(logger, SEARCH_ALPHAFOLD, node=SEARCH_ALPHAFOLD) as outcome:
                candidates = await self.capability.predicted(protein)
                outcome["candidates"] = len(candidates)
        except ProteinAgentError as exc:
            return degraded(
                SEARCH_ALPHAFOLD,
                failure_code(exc, "ALPHAFOLD_UNAVAILABLE", "ALPHAFOLD_TIMEOUT"),
                exc,
                alphafold_candidates=[],
            )
        return executed(
            SEARCH_ALPHAFOLD,
            alphafold_candidates=candidates,
            evidence=[_evidence("AlphaFold DB", candidate) for candidate in candidates],
        )

    def select_structure(self, state: ProteinWorkflowState) -> dict[str, Any]:
        with log_stage(logger, SELECT_STRUCTURE, node=SELECT_STRUCTURE) as outcome:
            selected, alternatives = self.capability.select(
                state["task"], state["valid_pdb_candidates"], state["alphafold_candidates"]
            )
            outcome["selected"] = selected.external_id if selected else None

        if selected is None:
            return executed(
                SELECT_STRUCTURE,
                selected_structure=None,
                alternative_structures=[],
                current_status=AnalysisStatus.no_structure_found,
                warnings=["NO_STRUCTURE_FOUND: neither RCSB PDB nor AlphaFold returned a usable model."],
            )

        warnings = []
        if selected.structure_type == "PREDICTED":
            warnings.append(
                "ALPHAFOLD_FALLBACK: no experimental structure qualified; the selected model is predicted."
            )
        return executed(
            SELECT_STRUCTURE,
            selected_structure=selected,
            alternative_structures=alternatives,
            warnings=warnings,
        )
