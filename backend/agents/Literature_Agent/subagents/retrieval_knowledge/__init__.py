"""Retrieval & Knowledge Processing - a leaf agent under Knowledge Discovery.

Only the adapter is exported. `finalagenttt.py` (the engine) is deliberately
not imported here: it pulls sentence-transformers, scikit-learn and a Qdrant
client, and this package sits on the Literature Agent's import chain.
"""
from .agent import DEFAULT_LIMIT, is_configured, search_literature

__all__ = ["search_literature", "is_configured", "DEFAULT_LIMIT"]
