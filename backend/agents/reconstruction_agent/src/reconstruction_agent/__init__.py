"""Umbrella Reconstruction Agent.

Reconstructs unresolved regions of incomplete animal genomes from homologous
reference sequence, and explains the evidence behind every proposed base.

The public surface is deliberately small - the API layer and the service - so
that internals can move without breaking callers.
"""
from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__", "create_app", "ReconstructionService"]


def __getattr__(name: str) -> object:
    """Expose the app factory and service without importing FastAPI eagerly.

    Keeps `import reconstruction_agent` cheap for consumers that only want the
    domain layer - the tests, and anything driving the agent in-process.
    """
    if name == "create_app":
        from .api.app import create_app

        return create_app
    if name == "ReconstructionService":
        from .application.reconstruction_service import ReconstructionService

        return ReconstructionService
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
