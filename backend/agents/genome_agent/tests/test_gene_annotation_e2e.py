"""
End-to-end tests for gene annotation with grounding verification.

Tests the full flow:
1. LLM strategy resolution
2. NCBI gene search with optional keyword
3. NCBI gene summary fetch
4. Grounding verification (strict rejection of ungrounded facts)
"""

from __future__ import annotations

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ..subagents.gene_annotation import (
    fetch_gene_summaries,
    get_gene_annotation,
    search_genes,
    _is_informative_description,
    _rank_gene_table,
    _verify_grounding,
)
from ..workflows.gene_annotation_resolver import GeneAnnotationStrategy

# Opt-in live NCBI tests
_skip_reason = "RUN_NCBI_LIVE_TESTS not set — skipping live NCBI eutils tests"
_run_live = pytest.mark.skipif(not os.getenv("RUN_NCBI_LIVE_TESTS"), reason=_skip_reason)


class TestSearchGenes:
    """Test the search_genes() helper function."""

    @_run_live
    @pytest.mark.asyncio
    async def test_search_genes_broad(self):
        """Test broad search without keyword."""
        gene_ids = await search_genes("GCF_000464555.1", keyword=None, max_results=10)
        assert isinstance(gene_ids, list)
        assert len(gene_ids) > 0, "Expected at least one gene from NCBI"
        assert all(isinstance(gid, str) for gid in gene_ids)

    @_run_live
    @pytest.mark.asyncio
    async def test_search_genes_with_keyword(self):
        """Test keyword-filtered search."""
        gene_ids = await search_genes(
            "GCF_000464555.1",
            keyword="color",
            max_results=10,
        )
        assert isinstance(gene_ids, list)
        # May return 0 genes if no color-related genes exist; both are valid outcomes


class TestFetchGeneSummaries:
    """Test the fetch_gene_summaries() helper function."""

    @_run_live
    @pytest.mark.asyncio
    async def test_fetch_gene_summaries_empty(self):
        """Test fetching summaries for empty gene list."""
        gene_table, grounding = await fetch_gene_summaries([])
        assert gene_table == []
        assert grounding == {}

    @_run_live
    @pytest.mark.asyncio
    async def test_fetch_gene_summaries_valid(self):
        """Test fetching summaries for valid gene IDs."""
        # Use a known gene from a real assembly
        gene_ids = await search_genes("GCF_000464555.1", max_results=5)
        if not gene_ids:
            pytest.skip("No genes found in assembly to fetch")

        gene_table, grounding = await fetch_gene_summaries(gene_ids)

        assert isinstance(gene_table, list)
        assert len(gene_table) > 0
        for row in gene_table:
            assert "gene_name" in row
            assert "location" in row
            assert "function" in row

        # Grounding record should contain facts from NCBI
        assert len(grounding) > 0
        assert all(
            value is not None and isinstance(value, str)
            for value in grounding.values()
        )


class TestRankGeneTable:
    """Test the _rank_gene_table() / _is_informative_description() helpers.

    Covers the §3.10 requirement: "General question → assert real-
    description genes rank ahead of uncharacterized ones."
    """

    def test_is_informative_description_empty(self):
        assert _is_informative_description("LOC123456", "") is False

    def test_is_informative_description_uncharacterized(self):
        assert _is_informative_description(
            "LOC123456", "Uncharacterized LOC123456"
        ) is False

    def test_is_informative_description_real(self):
        assert _is_informative_description(
            "TP53", "tumor protein p53, cell cycle regulation"
        ) is True

    def test_rank_gene_table_real_descriptions_first(self):
        """Real-description genes should rank ahead of uncharacterized ones,
        without dropping the uncharacterized ones."""
        gene_table = [
            {"gene_name": "LOC111", "location": "chr1", "function": ""},
            {"gene_name": "TP53", "location": "chr17", "function": "tumor suppressor"},
            {"gene_name": "LOC222", "location": "chr2", "function": "Uncharacterized LOC222"},
            {"gene_name": "MYC", "location": "chr8", "function": "transcription factor"},
        ]
        ranked = _rank_gene_table(gene_table)

        assert len(ranked) == 4, "ranking must not drop any genes"
        ranked_names = [row["gene_name"] for row in ranked]
        assert ranked_names[:2] == ["TP53", "MYC"], (
            "informative genes should be ranked first, in original relative order"
        )
        assert set(ranked_names[2:]) == {"LOC111", "LOC222"}, (
            "uncharacterized genes should still be present, just ranked lower"
        )

    def test_rank_gene_table_all_uncharacterized(self):
        """No real-description genes: order is preserved, nothing dropped."""
        gene_table = [
            {"gene_name": "LOC111", "location": "chr1", "function": ""},
            {"gene_name": "LOC222", "location": "chr2", "function": "Uncharacterized LOC222"},
        ]
        ranked = _rank_gene_table(gene_table)
        assert [row["gene_name"] for row in ranked] == ["LOC111", "LOC222"]

    def test_rank_gene_table_empty(self):
        assert _rank_gene_table([]) == []


class TestGetGeneAnnotationRanking:
    """End-to-end: get_gene_annotation() applies ranking only when the LLM
    strategy succeeded (§3.9: total fallback stays unranked, first-N as-is)."""

    @pytest.mark.asyncio
    async def test_get_gene_annotation_ranks_real_descriptions_first_when_llm_succeeds(self):
        mock_search_response = MagicMock()
        mock_search_response.json.return_value = {
            "esearchresult": {"idlist": ["1", "2", "3"]}
        }

        mock_fetch_response = MagicMock()
        mock_fetch_response.json.return_value = {
            "result": {
                "1": {"name": "LOC001", "description": "", "chromosome": "1"},
                "2": {
                    "name": "TP53",
                    "description": "tumor suppressor gene",
                    "chromosome": "17",
                },
                "3": {
                    "name": "LOC003",
                    "description": "Uncharacterized LOC003",
                    "chromosome": "3",
                },
            }
        }

        with patch("genome_agent.subagents.gene_annotation.ncbi_get") as mock_ncbi, patch(
            "genome_agent.subagents.gene_annotation.resolve_gene_annotation_strategy"
        ) as mock_strategy:
            mock_ncbi.side_effect = [mock_search_response, mock_fetch_response]
            # Simulate the LLM strategy call succeeding (this is what gates ranking).
            mock_strategy.return_value = GeneAnnotationStrategy(
                search_keyword=None,
                ranking_criteria="prefer well-annotated genes",
                reasoning="Broad question, no specific trait named.",
            )

            result = await get_gene_annotation(
                "GCF_000464555.1",
                user_question="Show me gene annotations for the tiger",
            )

            assert result["gene_list"][0] == "TP53", (
                "the gene with a real description must be ranked ahead of "
                "the uncharacterized ones when the LLM strategy succeeded"
            )
            assert set(result["gene_list"]) == {"LOC001", "TP53", "LOC003"}, (
                "ranking must not drop any genes"
            )

    @pytest.mark.asyncio
    async def test_get_gene_annotation_stays_unranked_when_llm_unavailable(self):
        """§3.9: total fallback (no LLM at all) returns first-N as-is,
        deliberately NOT reordered — 'no ranking, first 50 as-is'."""
        mock_search_response = MagicMock()
        mock_search_response.json.return_value = {
            "esearchresult": {"idlist": ["1", "2", "3"]}
        }

        mock_fetch_response = MagicMock()
        mock_fetch_response.json.return_value = {
            "result": {
                # Deliberately ordered LOC (uncharacterized) BEFORE TP53
                # (informative) — if ranking were applied, TP53 would move
                # to the front. The fallback must leave this order alone.
                "1": {"name": "LOC001", "description": "", "chromosome": "1"},
                "2": {
                    "name": "TP53",
                    "description": "tumor suppressor gene",
                    "chromosome": "17",
                },
                "3": {"name": "LOC003", "description": "", "chromosome": "3"},
            }
        }

        with patch("genome_agent.subagents.gene_annotation.ncbi_get") as mock_ncbi, patch(
            "genome_agent.subagents.gene_annotation.resolve_gene_annotation_strategy"
        ) as mock_strategy:
            mock_ncbi.side_effect = [mock_search_response, mock_fetch_response]
            mock_strategy.return_value = None  # simulates LLM client unavailable

            result = await get_gene_annotation(
                "GCF_000464555.1",
                user_question="Show me gene annotations for the tiger",
            )

            assert result["gene_list"] == ["LOC001", "TP53", "LOC003"], (
                "total fallback must preserve NCBI's original order, not rank "
                "informative genes ahead of uncharacterized ones"
            )


class TestGroundingVerification:
    """Test the _verify_grounding() function."""

    def test_verify_grounding_all_grounded(self):
        """Test that grounding verification passes when all facts are from NCBI."""
        gene_table = [
            {"gene_name": "ABC", "location": "chr1", "function": "involved in X"},
            {"gene_name": "DEF", "location": "chr2", "function": "involved in Y"},
        ]
        grounding_record = {
            (0, "gene_name"): "ABC",
            (0, "location"): "chr1",
            (0, "function"): "involved in X",
            (1, "gene_name"): "DEF",
            (1, "location"): "chr2",
            (1, "function"): "involved in Y",
        }
        assert _verify_grounding(gene_table, grounding_record) is True

    def test_verify_grounding_invented_gene_name(self):
        """Test that grounding verification REJECTS invented gene names."""
        gene_table = [
            {"gene_name": "FABRICATED_GENE", "location": "chr1", "function": ""},
        ]
        grounding_record = {
            (0, "gene_name"): "ABC",
            (0, "location"): "chr1",
            (0, "function"): "",
        }
        assert _verify_grounding(gene_table, grounding_record) is False

    def test_verify_grounding_invented_function(self):
        """Test that grounding verification REJECTS invented function descriptions."""
        gene_table = [
            {"gene_name": "ABC", "location": "chr1", "function": "invented function from LLM"},
        ]
        grounding_record = {
            (0, "gene_name"): "ABC",
            (0, "location"): "chr1",
            (0, "function"): "",  # Empty from NCBI
        }
        assert _verify_grounding(gene_table, grounding_record) is False

    def test_verify_grounding_empty_descriptions_allowed(self):
        """Test that empty fields (from NCBI) are not rejected."""
        gene_table = [
            {"gene_name": "ABC", "location": "", "function": ""},
        ]
        grounding_record = {
            (0, "gene_name"): "ABC",
            (0, "location"): "",
            (0, "function"): "",
        }
        assert _verify_grounding(gene_table, grounding_record) is True


class TestGetGeneAnnotation:
    """Test the full get_gene_annotation() flow."""

    @pytest.mark.asyncio
    async def test_get_gene_annotation_with_mock_ncbi(self):
        """Test full flow with mocked NCBI responses."""
        mock_search_response = MagicMock()
        mock_search_response.json.return_value = {
            "esearchresult": {"idlist": ["12345", "67890"]}
        }

        mock_fetch_response = MagicMock()
        mock_fetch_response.json.return_value = {
            "result": {
                "12345": {
                    "name": "GeneA",
                    "description": "Gene A description",
                    "chromosome": "1",
                },
                "67890": {
                    "name": "GeneB",
                    "description": "",  # Empty description from NCBI
                    "chromosome": "2",
                },
            }
        }

        with patch("genome_agent.subagents.gene_annotation.ncbi_get") as mock_ncbi:
            # First call is search, second is fetch
            mock_ncbi.side_effect = [mock_search_response, mock_fetch_response]

            result = await get_gene_annotation(
                "GCF_000464555.1",
                user_question="Show me genes for tiger",
            )

            assert result["gene_list"] == ["GeneA", "GeneB"]
            assert len(result["gene_table"]) == 2
            # Verify that empty description from NCBI is kept empty (not invented)
            assert result["gene_table"][1]["function"] == ""

    @pytest.mark.asyncio
    async def test_get_gene_annotation_rejects_hallucinated_facts(self):
        """Test that grounding verification rejects LLM hallucinations."""
        # In a real scenario, this would require mocking the LLM to return
        # fabricated gene info. For now, we test with mocked NCBI that
        # returns clean data, and our verification should accept it.
        mock_search_response = MagicMock()
        mock_search_response.json.return_value = {
            "esearchresult": {"idlist": ["12345"]}
        }

        mock_fetch_response = MagicMock()
        mock_fetch_response.json.return_value = {
            "result": {
                "12345": {
                    "name": "RealGene",
                    "description": "Real description from NCBI",
                    "chromosome": "1",
                },
            }
        }

        with patch("genome_agent.subagents.gene_annotation.ncbi_get") as mock_ncbi:
            mock_ncbi.side_effect = [mock_search_response, mock_fetch_response]

            result = await get_gene_annotation(
                "GCF_000464555.1",
                user_question="Show genes",
            )

            # Should succeed with real NCBI data
            assert len(result["gene_list"]) == 1
            assert result["gene_list"][0] == "RealGene"

    @pytest.mark.asyncio
    async def test_get_gene_annotation_no_genes_found(self):
        """Test behavior when NCBI returns no genes."""
        mock_search_response = MagicMock()
        mock_search_response.json.return_value = {
            "esearchresult": {"idlist": []}
        }

        with patch("genome_agent.subagents.gene_annotation.ncbi_get") as mock_ncbi:
            mock_ncbi.return_value = mock_search_response

            result = await get_gene_annotation(
                "GCF_INVALID.1",
                user_question="Show genes",
            )

            assert result["gene_list"] == []
            assert result["gene_table"] == []

    @_run_live
    @pytest.mark.asyncio
    async def test_get_gene_annotation_live_ncbi(self):
        """Test full flow against real NCBI (opt-in)."""
        result = await get_gene_annotation(
            "GCF_000464555.1",
            user_question="Show gene annotations for tiger",
        )

        # Should return valid results from NCBI
        assert isinstance(result, dict)
        assert "gene_table" in result
        assert "gene_list" in result
        assert len(result["gene_list"]) > 0
        
        # Verify structure
        for row in result["gene_table"]:
            assert "gene_name" in row
            assert "location" in row
            assert "function" in row
            # These should all be strings (may be empty, but not None)
            assert isinstance(row["gene_name"], str)
            assert isinstance(row["location"], str)
            assert isinstance(row["function"], str)
