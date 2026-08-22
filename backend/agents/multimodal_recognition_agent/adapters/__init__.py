"""Adapters: the boundary between the workflow and everything outside it.

Each adapter is a Protocol plus at least one implementation. The workflow only
ever sees the Protocol, which is what lets the mock BioCLIP-2 classifier be
swapped for the real one, and the mocked GBIF/NCBI sources for live ones,
without the workflow changing at all.

There is no retrieval adapter and no vector-store client here, and there is not
meant to be one. This agent classifies an image into taxonomic labels; it runs
no nearest-neighbour search and owns no reference-image corpus.
"""
