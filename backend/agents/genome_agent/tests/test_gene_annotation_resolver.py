"""
Tests for gene_annotation_resolver.py (LLM strategy selection).
"""

from __future__ import annotations

import pytest

from ..workflows.gene_annotation_resolver import (
    resolve_gene_annotation_strategy_fallback,
)


def test_resolve_strategy_fallback_general_question():
    """Test broad search for general gene question."""
    strategy = resolve_gene_annotation_strategy_fallback(
        "Show me all gene annotations for the tiger."
    )
    assert strategy is not None
    assert strategy.search_keyword is None or strategy.search_keyword == ""
    assert strategy.reasoning == "Keyword fallback"


def test_resolve_strategy_fallback_color():
    """Test trait-specific search for color-related genes."""
    strategy = resolve_gene_annotation_strategy_fallback(
        "Which genes affect fur color in tigers?"
    )
    assert strategy is not None
    # Should extract a color-related keyword
    assert strategy.search_keyword is not None
    assert "color" in strategy.search_keyword.lower() or "pigment" in strategy.search_keyword.lower()


def test_resolve_strategy_fallback_behavior():
    """Test trait-specific search for behavior genes."""
    strategy = resolve_gene_annotation_strategy_fallback(
        "What genes control social behavior?"
    )
    assert strategy is not None
    assert strategy.search_keyword is not None


def test_resolve_strategy_fallback_metabolism():
    """Test trait-specific search for metabolism genes."""
    strategy = resolve_gene_annotation_strategy_fallback(
        "Show me genes related to fat metabolism."
    )
    assert strategy is not None
    assert strategy.search_keyword is not None


def test_resolve_strategy_fallback_always_returns():
    """Test that fallback always returns valid strategy."""
    strategy = resolve_gene_annotation_strategy_fallback(
        "xyzzy 12345 random gibberish"
    )
    assert strategy is not None
    assert strategy.reasoning == "Keyword fallback"
