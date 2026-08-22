"""Re-export schema symbols so `genome_agent.schemas` (the package) works the
same way as `genome_agent.schema` (the top-level module).

Without this file, `schemas/` was an implicit namespace package: importing
`genome_agent.schemas.common` worked, but `from genome_agent.schemas import
AgentStatus` did not, because nothing re-exported the name at the package
level. Tests and any future caller should be able to use either the
top-level `schema` module or the `schemas` package interchangeably.
"""

from .common import AgentRequest, AgentResult, AgentStatus
from .inputs import (
    GeneAnnotationRequest,
    GenomeMetadataRequest,
    SpeciesResolverRequest,
    VisualizationRequest,
)
from .outputs import (
    GeneAnnotationOutput,
    GenomeAgentOutput,
    GenomeMetadataOutput,
    SpeciesResolverOutput,
    VisualizationOutput,
)

__all__ = [
    "AgentStatus",
    "AgentRequest",
    "AgentResult",
    "SpeciesResolverRequest",
    "SpeciesResolverOutput",
    "GenomeMetadataRequest",
    "GenomeMetadataOutput",
    "GeneAnnotationRequest",
    "GeneAnnotationOutput",
    "VisualizationRequest",
    "VisualizationOutput",
    "GenomeAgentOutput",
]
