from __future__ import annotations

import pytest

from ..workflows.query_router import route_query, route_query_fallback


def test_route_query_genome_size():
    decision = route_query_fallback("What is the genome size of the tiger?")
    assert decision.needs_metadata is True
    assert decision.needs_annotation is False
    assert decision.visualization_scope == "chromosome_map"


def test_route_query_gene_annotation():
    decision = route_query_fallback("Show gene annotations for the house mouse.")
    assert decision.needs_metadata is False
    assert decision.needs_annotation is True
    assert decision.visualization_scope == "none"


def test_route_query_protein_structure():
    decision = route_query_fallback("Predict the 3D protein structure for the woolly mammoth.")
    assert decision.needs_metadata is False
    assert decision.needs_annotation is True
    assert decision.visualization_scope == "protein_structure"


def test_route_query_chromosome_map():
    decision = route_query_fallback("Show me a chromosome map of the asian elephant.")
    assert decision.needs_metadata is True
    assert decision.needs_annotation is False
    assert decision.visualization_scope == "chromosome_map"


def test_route_query_size_comparison():
    decision = route_query_fallback("Compare the genome size of tiger and lion.")
    assert decision.needs_metadata is True
    assert decision.visualization_scope == "size_comparison"


def test_route_query_fallback_always_returns():
    decision = route_query_fallback("random gibberish xyzzy")
    assert decision is not None
    assert decision.needs_species_resolution is True
    assert decision.reasoning == "Fallback keyword match"
