"""Domain models: the vocabulary the whole agent reasons in.

These are frozen Pydantic models. Immutability is deliberate - they travel
through a LangGraph state that merges concurrent updates, and a value that can
be mutated in place is a value that can be changed by one branch while another
is reading it.

Nothing in this package imports from `integrations` or `services`. The domain
describes biology and results; it does not know that EMBL-EBI or NCBI exist.
"""

from __future__ import annotations

from reconstruction_agent.domain.models.alignment import (
    AlignedRow,
    Alignment,
    AlignmentSupport,
    ReferenceFill,
)
from reconstruction_agent.domain.models.candidate import (
    Candidate,
    CandidateScores,
    ValidationVerdict,
)
from reconstruction_agent.domain.models.homology import HomologHit, HomologySearchOutcome
from reconstruction_agent.domain.models.result import (
    AlignmentEvidence,
    Evo2Evidence,
    GapEvidence,
    GapReconstruction,
    HomologyEvidence,
    Provenance,
    ReconstructionResult,
)
from reconstruction_agent.domain.models.sequence import (
    DNA_ALPHABET,
    AssemblyMetadata,
    Gap,
    GapContext,
    SequenceRecord,
)
from reconstruction_agent.domain.models.taxonomy import TargetProfile, TaxonNode

__all__ = [
    "DNA_ALPHABET",
    "AlignedRow",
    "Alignment",
    "AlignmentEvidence",
    "AlignmentSupport",
    "AssemblyMetadata",
    "Candidate",
    "CandidateScores",
    "Evo2Evidence",
    "Gap",
    "GapContext",
    "GapEvidence",
    "GapReconstruction",
    "HomologHit",
    "HomologyEvidence",
    "HomologySearchOutcome",
    "Provenance",
    "ReconstructionResult",
    "ReferenceFill",
    "SequenceRecord",
    "TargetProfile",
    "TaxonNode",
    "ValidationVerdict",
]
