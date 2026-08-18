"""Input/output shapes for the NCBI tool."""
from __future__ import annotations

from pydantic import Field

from ...contracts.output import EvidenceItem
from ...domain.models import Reference
from ..contracts import ToolInput, ToolOutput


class NCBISearchInput(ToolInput):
    """Find and optionally fetch reference sequences from NCBI."""

    term: str = Field(description="Entrez query, e.g. 'Loxodonta africana[Organism] AND MT-CO1'.")
    organisms: list[str] = Field(
        default_factory=list, description="Restrict to these organisms, OR-ed into the query."
    )
    database: str = "nuccore"
    limit: int = Field(default=10, ge=1, le=100)
    # Fetching residues costs a second round trip; the planner skips it when it
    # only needs to know what exists.
    fetch_sequences: bool = True


class NCBISearchOutput(ToolOutput):
    references: list[Reference] = Field(
        default_factory=list, description="Residues are present only when fetched."
    )
    evidence: list[EvidenceItem] = Field(default_factory=list)
    total_found: int = 0
