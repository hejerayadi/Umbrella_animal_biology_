"""Scientific Analysis - a leaf agent under Knowledge Discovery.

Only the adapter is exported, same convention as retrieval_knowledge/__init__.py:
the engine pulls sentence-transformers, a Qdrant client and an OpenAI client,
and this package sits on the Literature Agent's import chain.
"""
from .agent import run_scientific_analysis, is_configured

__all__ = ["run_scientific_analysis", "is_configured"]
