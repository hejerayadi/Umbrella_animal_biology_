"""
End-to-end tests for visualization with grounding verification.

Tests the full flow:
1. LLM reference species selection (for size_comparison)
2. resolve_reference_species() -> resolve_species() + get_genome_metadata()
3. Pure SVG rendering
4. Grounding verification (all genome sizes from NCBI)
5. Deterministic protein_structure delegation
"""

from __future__ import annotations

import os
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ..subagents.visualization import (
    generate_visualization,
    resolve_reference_species,
)
from ..subagents.visualization_render import (
    render_chromosome_map,
    render_size_comparison,
)

# Opt-in live NCBI tests
_skip_reason = "RUN_NCBI_LIVE_TESTS not set — skipping live NCBI eutils tests"
_run_live = pytest.mark.skipif(not os.getenv("RUN_NCBI_LIVE_TESTS"), reason=_skip_reason)


class TestRenderChromosomeMap:
    """Test the pure render_chromosome_map() function."""

    def test_render_chromosome_map_empty(self):
        """Test rendering with empty gene table."""
        svg = render_chromosome_map([])
        assert isinstance(svg, bytes)
        assert b"No gene data available" in svg

    def test_render_chromosome_map_with_genes(self):
        """Test rendering with gene data."""
        gene_table = [
            {"gene_name": "ABC", "location": "chr1", "function": "Function A"},
            {"gene_name": "DEF", "location": "chr2", "function": "Function B"},
        ]
        svg = render_chromosome_map(gene_table)
        assert isinstance(svg, bytes)
        assert b"ABC" in svg
        assert b"DEF" in svg
        assert b"chr1" in svg
        assert b"chr2" in svg

    def test_render_chromosome_map_with_highlight(self):
        """Test that highlight_gene is reflected in SVG."""
        gene_table = [
            {"gene_name": "GeneA", "location": "chr1", "function": "F1"},
            {"gene_name": "GeneB", "location": "chr2", "function": "F2"},
        ]
        svg = render_chromosome_map(gene_table, highlight_gene="GeneA")
        assert isinstance(svg, bytes)
        # Should contain SVG with both genes
        assert b"GeneA" in svg
        assert b"GeneB" in svg
        # The highlight color (#e8622c) should appear for GeneA
        assert b"e8622c" in svg


class TestRenderSizeComparison:
    """Test the pure render_size_comparison() function."""

    def test_render_size_comparison_empty(self):
        """Test rendering with empty rows."""
        svg = render_size_comparison([])
        assert isinstance(svg, bytes)
        assert b"No comparison data" in svg

    def test_render_size_comparison_single_species(self):
        """Test rendering with one species."""
        rows = [
            {
                "common_name": "human",
                "assembly_id": "GCF_human",
                "genome_size_bp": 3_200_000_000,
            }
        ]
        svg = render_size_comparison(rows, current_assembly_id="GCF_human")
        assert isinstance(svg, bytes)
        assert b"human" in svg
        assert b"3.20 Gb" in svg
        assert b"e8622c" in svg  # Highlight color for queried species

    def test_render_size_comparison_multiple_species(self):
        """Test rendering with multiple species."""
        rows = [
            {
                "common_name": "tiger",
                "assembly_id": "GCF_tiger",
                "genome_size_bp": 2_700_000_000,
            },
            {
                "common_name": "human",
                "assembly_id": "GCF_human",
                "genome_size_bp": 3_200_000_000,
            },
            {
                "common_name": "mouse",
                "assembly_id": "GCF_mouse",
                "genome_size_bp": 2_700_000_000,
            },
        ]
        svg = render_size_comparison(rows, current_assembly_id="GCF_tiger")
        assert isinstance(svg, bytes)
        assert b"tiger" in svg
        assert b"human" in svg
        assert b"mouse" in svg
        # Should have queried label
        assert b"(queried)" in svg


class TestResolveReferenceSpecies:
    """Test the resolve_reference_species() helper."""

    @pytest.mark.asyncio
    async def test_resolve_reference_species_empty(self):
        """Test with empty candidate list."""
        result = await resolve_reference_species([])
        assert result == []

    @pytest.mark.asyncio
    async def test_resolve_reference_species_with_mock(self):
        """Test reference species resolution with mocked NCBI."""
        mock_species_result = {
            "assembly_id": "GCF_mouse",
            "common_name": "house mouse",
            "scientific_name": "Mus musculus",
            "confidence": 0.95,
        }
        mock_metadata_result = {
            "genome_size_bp": 2_700_000_000,
            "chromosome_count": 20,
            "karyotype": None,
            "assembly_level": "Complete Genome",
        }

        with patch(
            "genome_agent.subagents.visualization.resolve_species"
        ) as mock_rs, patch(
            "genome_agent.subagents.visualization.get_genome_metadata"
        ) as mock_gm:
            mock_rs.return_value = mock_species_result
            mock_gm.return_value = mock_metadata_result

            result = await resolve_reference_species(["house mouse"])

            assert len(result) == 1
            assert result[0]["common_name"] == "house mouse"
            assert result[0]["genome_size_bp"] == 2_700_000_000
            assert result[0]["assembly_id"] == "GCF_mouse"

    @pytest.mark.asyncio
    async def test_resolve_reference_species_partial_failure(self):
        """Test that failed references are dropped silently."""
        mock_species_result = {
            "assembly_id": "GCF_human",
            "common_name": "human",
            "scientific_name": "Homo sapiens",
            "confidence": 0.99,
        }
        mock_metadata_result = {
            "genome_size_bp": 3_200_000_000,
            "chromosome_count": 24,
            "karyotype": None,
            "assembly_level": "Complete Genome",
        }

        with patch(
            "genome_agent.subagents.visualization.resolve_species"
        ) as mock_rs, patch(
            "genome_agent.subagents.visualization.get_genome_metadata"
        ) as mock_gm:
            # First call returns valid result, second raises Exception
            mock_rs.side_effect = [mock_species_result, Exception("Not found")]
            mock_gm.return_value = mock_metadata_result

            result = await resolve_reference_species(["human", "alien"])

            # Should only have the successfully resolved species
            assert len(result) == 1
            assert result[0]["common_name"] == "human"


class TestGenerateVisualization:
    """Test the full generate_visualization() function."""

    @pytest.mark.asyncio
    async def test_generate_visualization_chromosome_map_empty(self):
        """Test chromosome_map with empty gene table."""
        result = await generate_visualization(
            scope="chromosome_map",
            gene_table=[],
        )
        assert result["status"] == "COMPLETED"
        assert result["chart_data"] is None
        assert "No gene data available" in result.get("note", "")

    @pytest.mark.asyncio
    async def test_generate_visualization_chromosome_map_with_genes(self):
        """Test chromosome_map with gene data."""
        gene_table = [
            {"gene_name": "ABC", "location": "chr1", "function": "Function A"},
        ]
        result = await generate_visualization(
            scope="chromosome_map",
            gene_table=gene_table,
        )
        assert result["status"] == "COMPLETED"
        assert result["chart_data"] is not None
        assert result["format"] == "svg"
        assert b"ABC" in result["chart_data"]

    @pytest.mark.asyncio
    async def test_generate_visualization_chromosome_map_highlight_from_question(self):
        """chromosome_map with a gene named in the question → highlight_gene
        is set from the question (§4.10). Mocks the LLM resolver directly
        since generate_visualization() has no LLM-availability parameter."""
        gene_table = [
            {"gene_name": "ABC", "location": "chr1", "function": "Function A"},
            {"gene_name": "Trp53", "location": "chr17", "function": "tumor suppressor"},
        ]
        with patch(
            "genome_agent.subagents.visualization.resolve_chromosome_highlight"
        ) as mock_resolve:
            mock_resolve.return_value = SimpleNamespace(
                highlight_gene="Trp53", reasoning="Question asks about Trp53."
            )
            result = await generate_visualization(
                scope="chromosome_map",
                gene_table=gene_table,
                user_question="Show me the chromosome map with Trp53 highlighted",
            )

        assert result["status"] == "COMPLETED"
        mock_resolve.assert_called_once()
        # highlight_gene came from the (mocked) LLM resolver, not invented locally
        called_question, called_candidates = mock_resolve.call_args[0]
        assert called_question == "Show me the chromosome map with Trp53 highlighted"
        assert set(called_candidates) == {"ABC", "Trp53"}

    @pytest.mark.asyncio
    async def test_generate_visualization_chromosome_map_highlight_fallback(self):
        """When the LLM resolver is unavailable, the deterministic fallback
        still finds a highlight_gene named in the question."""
        gene_table = [
            {"gene_name": "ABC", "location": "chr1", "function": "Function A"},
        ]
        with patch(
            "genome_agent.subagents.visualization.resolve_chromosome_highlight"
        ) as mock_resolve:
            mock_resolve.return_value = None  # simulate LLM unavailable
            result = await generate_visualization(
                scope="chromosome_map",
                gene_table=gene_table,
                user_question="Tell me about ABC please",
            )

        assert result["status"] == "COMPLETED"
        assert result["chart_data"] is not None
        # Fallback heuristic should still have picked ABC out of the question
        assert b"e8622c" in result["chart_data"] or b"ABC" in result["chart_data"]

    @pytest.mark.asyncio
    async def test_generate_visualization_protein_structure(self):
        """Test that protein_structure is deterministic (no LLM call)."""
        result = await generate_visualization(
            scope="protein_structure",
            assembly_id="GCF_xxx",
        )
        # Should be deterministic delegation
        assert result["status"] == "NEEDS_AGENT"
        assert result["target_agent"] is None
        assert result["prompt_to_target_agent"] is not None
        assert b"protein" not in (result.get("chart_data") or b"")

    @pytest.mark.asyncio
    async def test_generate_visualization_protein_structure_with_context(self):
        """Test protein_structure handoff includes context."""
        result = await generate_visualization(
            scope="protein_structure",
            user_question="Show protein structure for ABC gene",
            assembly_id="GCF_tiger",
        )
        assert result["status"] == "NEEDS_AGENT"
        # Handoff prompt should include user question and context
        assert "protein" in result["prompt_to_target_agent"].lower()
        assert "ABC" in result["prompt_to_target_agent"]

    @pytest.mark.asyncio
    async def test_generate_visualization_unknown_scope(self):
        """Test that unknown scope fails gracefully."""
        result = await generate_visualization(
            scope="unknown_visualization_type",
        )
        assert result["status"] == "FAILED"
        assert result["chart_data"] is None

    @pytest.mark.asyncio
    async def test_generate_visualization_size_comparison_with_mock(self):
        """Test size_comparison with mocked species resolution."""
        mock_species_result = {
            "assembly_id": "GCF_mouse",
            "common_name": "house mouse",
            "scientific_name": "Mus musculus",
            "confidence": 0.95,
        }
        mock_metadata_result = {
            "genome_size_bp": 2_700_000_000,
            "chromosome_count": 20,
            "karyotype": None,
            "assembly_level": "Complete Genome",
        }

        with patch(
            "genome_agent.subagents.visualization.resolve_visualization_references"
        ) as mock_resolver, patch(
            "genome_agent.subagents.visualization.resolve_reference_species"
        ) as mock_resolve_refs:
            # LLM picks references
            mock_resolver.return_value = MagicMock(
                reference_species=["house mouse", "chicken"],
                reasoning="Good comparisons",
            )
            # Reference resolution returns grounded data
            mock_resolve_refs.return_value = [
                {
                    "assembly_id": "GCF_mouse",
                    "common_name": "house mouse",
                    "scientific_name": "Mus musculus",
                    "genome_size_bp": 2_700_000_000,
                },
            ]

            result = await generate_visualization(
                scope="size_comparison",
                genome_size_bp=3_200_000_000,
                assembly_id="GCF_tiger",
                common_name="tiger",
                user_question="Compare tiger with other animals",
            )

            assert result["status"] == "COMPLETED"
            assert result["chart_data"] is not None
            assert result["format"] == "svg"
            assert result["comparisons"] is not None
            assert len(result["comparisons"]) >= 1

    @pytest.mark.asyncio
    async def test_generate_visualization_size_comparison_no_refs(self):
        """Test size_comparison when no references are resolved."""
        with patch(
            "genome_agent.subagents.visualization.resolve_visualization_references"
        ) as mock_resolver, patch(
            "genome_agent.subagents.visualization.resolve_reference_species"
        ) as mock_resolve_refs:
            mock_resolver.return_value = MagicMock(
                reference_species=["fictional_species"],
                reasoning="Test",
            )
            mock_resolve_refs.return_value = []  # All references failed

            result = await generate_visualization(
                scope="size_comparison",
                genome_size_bp=3_200_000_000,
                assembly_id="GCF_tiger",
                common_name="tiger",
            )

            assert result["status"] == "COMPLETED"
            assert result["chart_data"] is not None
            # Should have the queried species even if references failed
            assert result["comparisons"] is not None

    @_run_live
    @pytest.mark.asyncio
    async def test_generate_visualization_live_ncbi(self):
        """Test full flow against real NCBI (opt-in)."""
        result = await generate_visualization(
            scope="size_comparison",
            genome_size_bp=2_700_000_000,
            assembly_id="GCF_000464555.1",
            common_name="tiger",
            user_question="Compare tiger genome size with other mammals",
        )

        assert isinstance(result, dict)
        assert result["status"] == "COMPLETED"
        assert result["format"] == "svg"
        assert result["chart_data"] is not None
        assert isinstance(result["chart_data"], bytes)
        assert b"tiger" in result["chart_data"]
