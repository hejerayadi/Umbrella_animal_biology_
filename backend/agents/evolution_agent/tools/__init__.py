"""Real bioinformatics tools for the Evolution Agent.

Tools:
    - MAFFT: multiple sequence alignment
    - IQ-TREE: phylogenetic tree building with ModelFinder + UFBoot

Pipeline:
    Sequences → MAFFT → aligned sequences → IQ-TREE → tree + support
"""

from .mafft import align as mafft_align
from .mafft import parse_aligned as mafft_parse
from .mafft import MAFFTError
from .iqtree import build_tree as iqtree_build
from .iqtree import IQTreeError

__all__ = [
    "mafft_align",
    "mafft_parse",
    "MAFFTError",
    "iqtree_build",
    "IQTreeError",
]
