"""The vocabulary the agent reasons in: sequences, gaps, references, candidates.

These are plain dataclasses with no I/O and no framework dependencies, so the
domain services can be tested without a network or an LLM.
"""
from domain.models.alignment import GAP_CHARACTER, AlignedPair, Alignment
from domain.models.candidate import Candidate
from domain.models.gap import Gap, GapContext
from domain.models.reference import Reference
from domain.models.sequence import IUPAC_NUCLEOTIDES, UNKNOWN_BASE, Sequence

__all__ = [
    "GAP_CHARACTER",
    "IUPAC_NUCLEOTIDES",
    "UNKNOWN_BASE",
    "AlignedPair",
    "Alignment",
    "Candidate",
    "Gap",
    "GapContext",
    "Reference",
    "Sequence",
]
