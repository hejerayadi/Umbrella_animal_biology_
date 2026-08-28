"""The agent's tool surface.

Assembling the registry lives here rather than in the API layer so that a test,
a script and the running service all build the same surface from the same
services. A tool that exists but is not registered is unreachable, and a
`ToolName` with no tool behind it is a plan the registry refuses - both are
worth catching in one place.
"""

from __future__ import annotations

from reconstruction_agent.integrations.mafft.client import MafftClient
from reconstruction_agent.services.candidate.candidate_builder import CandidateBuilder
from reconstruction_agent.services.homology.homology_service import HomologyService
from reconstruction_agent.services.plausibility.evo2_service import Evo2Service
from reconstruction_agent.services.scoring.confidence_engine import ConfidenceEngine
from reconstruction_agent.services.sequence.sequence_service import SequenceService
from reconstruction_agent.services.taxonomy.taxonomy_service import TaxonomyService
from reconstruction_agent.tools.alignment import AlignHomologsTool, AnalyzeAlignmentTool
from reconstruction_agent.tools.base import Tool, ToolOutcome, ToolRecord
from reconstruction_agent.tools.candidate import GenerateCandidatesTool, ScoreCandidateTool
from reconstruction_agent.tools.finalize import FinalizeResultTool
from reconstruction_agent.tools.homology import GetHomologSequencesTool, SearchHomologsTool
from reconstruction_agent.tools.plausibility import EvaluateWithEvo2Tool
from reconstruction_agent.tools.reconstruction import ReconstructGapTool
from reconstruction_agent.tools.registry import ToolRegistry
from reconstruction_agent.tools.sequence import GetAssemblyMetadataTool, GetSequenceContextTool
from reconstruction_agent.tools.validation import ValidateCandidateTool

__all__ = [
    "Tool",
    "ToolOutcome",
    "ToolRecord",
    "ToolRegistry",
    "build_registry",
]


def build_registry(
    *,
    sequences: SequenceService,
    taxonomy: TaxonomyService,
    homology: HomologyService,
    mafft: MafftClient,
    candidates: CandidateBuilder,
    engine: ConfidenceEngine,
    evo2: Evo2Service,
) -> ToolRegistry:
    """Every tool the agent can invoke, built over the shared services."""
    return ToolRegistry(
        (
            GetSequenceContextTool(sequences),
            GetAssemblyMetadataTool(taxonomy),
            SearchHomologsTool(homology),
            GetHomologSequencesTool(sequences),
            AlignHomologsTool(mafft),
            AnalyzeAlignmentTool(),
            GenerateCandidatesTool(candidates),
            EvaluateWithEvo2Tool(evo2, engine),
            ScoreCandidateTool(engine),
            ValidateCandidateTool(engine),
            ReconstructGapTool(),
            FinalizeResultTool(engine),
        )
    )
