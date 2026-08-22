"""External evidence tools for the image generation agent."""

from .pdb import call_pdb
from .uniprot import call_uniprot
from .web_search import call_web_search

__all__ = ["call_pdb", "call_uniprot", "call_web_search"]
