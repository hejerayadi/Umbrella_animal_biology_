"""Adapters: the boundary between the workflow and everything outside it.

Each adapter is a Protocol plus at least one implementation. The workflow only
ever sees the Protocol, which is what lets the mock BioCLIP-2 provider be
swapped for the real one, and the local fixture retriever for the real Qdrant
collection, without the workflow changing at all.
"""
