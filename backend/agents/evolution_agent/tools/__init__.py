"""Real bioinformatics tools for the Evolution Agent.

Tools:
    - MAFFT: multiple sequence alignment
    - IQ-TREE: phylogenetic tree building with ModelFinder + UFBoot

Pipeline:
    Sequences → MAFFT → aligned sequences → IQ-TREE → tree + support
"""

from .mafft import align as mafft_align
from .mafft import MAFFTError
from .iqtree import build_tree as iqtree_build
from .iqtree import IQTreeError

__all__ = [
    "mafft_align",
    "MAFFTError",
    "iqtree_build",
    "IQTreeError",
]
