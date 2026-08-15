"""
Tests for visualization_resolver.py (LLM reference species selection).
"""

from __future__ import annotations

import pytest

from ..workflows.visualization_resolver import (
    resolve_visualization_references_fallback,
)


def test_resolve_visualization_references_fallback():
    """Test that fallback returns sensible default reference species."""
    result = resolve_visualization_references_fallback()
    assert result is not None
    assert isinstance(result.reference_species, list)
    assert len(result.reference_species) == 4
    assert "human" in result.reference_species
    assert "house mouse" in result.reference_species
    assert "chicken" in result.reference_species
    assert "zebrafish" in result.reference_species
    assert result.reasoning == "Fallback: using well-known model organisms"


def test_resolve_visualization_references_fallback_with_species():
    """Test fallback with species argument (still returns default)."""
    result = resolve_visualization_references_fallback("tiger")
    assert result is not None
    assert len(result.reference_species) == 4


def test_resolve_visualization_references_fallback_always_returns():
    """Test that fallback always returns valid result."""
    result = resolve_visualization_references_fallback("")
    assert result is not None
    assert hasattr(result, "reference_species")
    assert hasattr(result, "reasoning")
