"""Molecular Comparison sub-agent.

``MolecularComparisonMock`` is imported lazily: it is only used by the test
suite, and importing it eagerly would drag networkx into every consumer of
this package for no reason in production.
"""

from .logic import MolecularComparisonAgent

__all__ = ["MolecularComparisonAgent", "MolecularComparisonMock"]


def __getattr__(name: str):
    if name == "MolecularComparisonMock":
        from .mock import MolecularComparisonMock
        return MolecularComparisonMock
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
