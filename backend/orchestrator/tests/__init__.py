"""Tests for the Global Orchestrator.

A package so that `backend.orchestrator.*` resolves the same way here as it
does in the code under test: pytest walks up from a test file to the first
directory without an `__init__.py`, which makes the repository root - and
therefore the `backend` package - importable.
"""
