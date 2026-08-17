"""
Gene Annotation — real NCBI Gene eutils subagent with LLM-guided search.

Retrieves gene/feature annotation data for a resolved assembly.
Never cached — pass-through only, rebuilt from NCBI each request
(gene tables can be large, so this is intentionally not stored anywhere,
per the consolidated store-or-not matrix).

CRITICAL: Every fact in the final answer must be traceable to a tool
response in the same execution (the "grounding rule"). This module:
1. Calls NCBI to search for genes (optionally filtering by keyword)
2. Batches fetches gene summaries from NCBI
3. Verifies grounding: every gene name/location/function came from NCBI
4. Rejects any fact that LLM may have hallucinated
5. Ranks genes with real, informative descriptions ahead of
   "uncharacterized"/no-description ones (never drops the latter)

Output shape matches schemas.outputs.GeneAnnotationOutput exactly
(gene_table, gene_list).
"""

from __future__ import annotations

import asyncio
import logging

from ..workflows.gene_annotation_resolver import (
    resolve_gene_annotation_strategy,
    resolve_gene_annotation_strategy_fallback,
)
from ._ncbi_client import ncbi_get

logger = logging.getLogger(__name__)

_MAX_GENES = 50


async def search_genes(
    assembly_id: str,
    keyword: str | None = None,
    max_results: int = _MAX_GENES,
) -> list[str]:
    """
    Search NCBI Gene for genes in an assembly, optionally filtered by keyword.
    
    Args:
        assembly_id: e.g., "GCF_000464555.1"
        keyword: Optional trait/feature keyword (e.g., "color")
        max_results: Max genes to retrieve (default 50)
    
    Returns:
        List of NCBI gene IDs (strings)
    """
    if keyword:
        term = f"{assembly_id}[Assembly] AND {keyword}[Gene Name]"
    else:
        term = f"{assembly_id}[Assembly]"
    
    resp = await asyncio.to_thread(
        ncbi_get,
        {
            "path": "esearch.fcgi",
            "db": "gene",
            "term": term,
            "retmode": "json",
            "retmax": max_results,
        },
    )
    data = resp.json()
    gene_ids = data.get("esearchresult", {}).get("idlist", [])
    return gene_ids


async def fetch_gene_summaries(gene_ids: list[str]) -> tuple[list[dict], dict]:
    """
    Batch-fetch NCBI gene summaries for a list of gene IDs.
    
    Args:
        gene_ids: List of NCBI gene ID strings
    
    Returns:
        Tuple of (gene_table, grounding_record) where:
        - gene_table: list of dicts with keys gene_name, location, function
        - grounding_record: dict mapping (gene_id, field) -> value for verification
    """
    if not gene_ids:
        return [], {}
    
    resp = await asyncio.to_thread(
        ncbi_get,
        {
            "path": "esummary.fcgi",
            "db": "gene",
            "id": ",".join(gene_ids[:_MAX_GENES]),
            "retmode": "json",
        },
    )
    data = resp.json()
    results = data.get("result", {})
    
    gene_table = []
    grounding_record = {}
    
    for gene_id in gene_ids[:_MAX_GENES]:
        gene_info = results.get(gene_id, {})
        gene_name = gene_info.get("name") or gene_info.get("Name") or gene_id
        description = gene_info.get("description") or gene_info.get("Description") or ""
        chromosome = gene_info.get("chromosome") or gene_info.get("chromosomes") or ""
        location = f"{chromosome}" if chromosome else ""
        
        # Record grounding facts: what we received from NCBI
        grounding_record[(gene_id, "gene_name")] = gene_name
        grounding_record[(gene_id, "location")] = location
        grounding_record[(gene_id, "function")] = description
        
        gene_table.append(
            {
                "gene_name": gene_name,
                "location": location,
                "function": description,
            }
        )
    
    return gene_table, grounding_record


def _is_informative_description(gene_name: str, function: str) -> bool:
    """
    True if a gene's description is real/informative rather than a stand-in
    for "we don't know anything about this gene yet".

    NCBI marks unstudied genes in a couple of recognizable ways: an empty
    description field, or a description/name that is just "uncharacterized
    LOC######" (no real biology asserted). Anything else counts as
    informative, even if terse.
    """
    if not function:
        return False
    lowered = function.strip().lower()
    if lowered.startswith("uncharacterized"):
        return False
    if gene_name.upper().startswith("LOC") and lowered.startswith(gene_name.lower()):
        # Some records echo the placeholder LOC id back as the "description".
        return False
    return True


def _rank_gene_table(
    gene_table: list[dict],
    ranking_criteria: str | None = None,
) -> list[dict]:
    """
    Rank genes with real, informative descriptions ahead of ones marked
    "uncharacterized" or with no description — without dropping the
    uncharacterized ones, per §3.3/§3.4/§3.8 of the design spec.

    This is a stable sort: genes keep their relative NCBI order within
    each bucket (informative vs. uncharacterized), so the LLM's or
    fallback's original search ordering is preserved as a tiebreaker.
    `ranking_criteria` is currently informational only (logged, not used
    to further reorder) — the one hard ranking rule the spec requires is
    "real description beats no description", which is deterministic and
    doesn't need an LLM call per gene.
    """
    if ranking_criteria:
        logger.info("[gene_annotation] ranking_criteria: %r", ranking_criteria)

    informative = [
        row for row in gene_table
        if _is_informative_description(row.get("gene_name", ""), row.get("function", ""))
    ]
    uncharacterized = [
        row for row in gene_table
        if not _is_informative_description(row.get("gene_name", ""), row.get("function", ""))
    ]
    return informative + uncharacterized


def _verify_grounding(
    gene_table: list[dict],
    grounding_record: dict,
) -> bool:
    """
    Verify that every fact in gene_table came from grounding_record.
    
    The grounding rule: no fact should be invented by LLM.
    Every gene_name, location, function must appear in what NCBI returned.
    
    Returns:
        True if all facts are grounded, False otherwise.
    """
    # Flatten grounding record into a set of facts
    grounded_facts = set(grounding_record.values())
    
    for row in gene_table:
        gene_name = row.get("gene_name", "")
        location = row.get("location", "")
        function = row.get("function", "")
        
        # Check that each field either came from NCBI or is empty
        if gene_name and gene_name not in grounded_facts:
            logger.warning(
                "GROUNDING FAILURE: gene_name %r not found in NCBI results",
                gene_name,
            )
            return False
        if location and location not in grounded_facts:
            logger.warning(
                "GROUNDING FAILURE: location %r not found in NCBI results",
                location,
            )
            return False
        if function and function not in grounded_facts:
            logger.warning(
                "GROUNDING FAILURE: function %r not found in NCBI results",
                function,
            )
            return False
    
    return True


async def get_gene_annotation(
    assembly_id: str,
    user_question: str = "",
) -> dict:
    """
    Real version of Gene Annotation with LLM-guided search strategy.
    
    Input:
        assembly_id (str): e.g., "GCF_000464555.1"
        user_question (str): Optional user question for context
    
    Output:
        dict matching GeneAnnotationOutput (gene_table, gene_list)
    
    Behavior:
        1. Ask LLM for search strategy (broad vs. trait-specific)
        2. Search NCBI Gene with optional keyword
        3. Fetch summaries for returned genes
        4. Verify grounding: all facts from NCBI, none invented
        5. Return results
    """
    # Step 1: Get search strategy from LLM
    llm_strategy = resolve_gene_annotation_strategy(user_question, assembly_id)
    used_llm_strategy = llm_strategy is not None
    strategy = llm_strategy or resolve_gene_annotation_strategy_fallback(user_question)
    
    logger.info(
        "[gene_annotation] strategy: keyword=%r, reasoning=%r",
        strategy.search_keyword,
        strategy.reasoning,
    )
    
    # Step 2: Search for genes with optional keyword
    try:
        gene_ids = await search_genes(
            assembly_id,
            keyword=strategy.search_keyword,
            max_results=_MAX_GENES,
        )
    except Exception as exc:
        logger.error("search_genes failed: %s", exc)
        return {
            "gene_table": [],
            "gene_list": [],
        }
    
    if not gene_ids:
        logger.info("[gene_annotation] no genes found for assembly %r", assembly_id)
        return {
            "gene_table": [],
            "gene_list": [],
        }
    
    # Step 3: Fetch summaries for genes
    try:
        gene_table, grounding_record = await fetch_gene_summaries(gene_ids)
    except Exception as exc:
        logger.error("fetch_gene_summaries failed: %s", exc)
        return {
            "gene_table": [],
            "gene_list": [],
        }
    
    if not gene_table:
        return {
            "gene_table": [],
            "gene_list": [],
        }
    
    # Step 4: Verify grounding before returning
    if not _verify_grounding(gene_table, grounding_record):
        logger.error(
            "GROUNDING VERIFICATION FAILED: some facts in gene_table are not from NCBI. "
            "Rejecting answer."
        )
        return {
            "gene_table": [],
            "gene_list": [],
        }
    
    # Step 5: Rank real-description genes ahead of uncharacterized ones —
    # but only when the LLM strategy actually succeeded. Per §3.9, the
    # total-fallback path (no LLM at all) stays deterministic and simple:
    # "no ranking, first 50 as-is — worse, but never broken." Ranking is
    # part of the LLM-guided path, not a universal post-process.
    if used_llm_strategy:
        gene_table = _rank_gene_table(gene_table, strategy.ranking_criteria)

    # Step 6: Build gene_list (must be exactly named for cross-agent contract)
    gene_list = [row["gene_name"] for row in gene_table]
    
    return {
        "gene_table": gene_table,
        "gene_list": gene_list,
    }


if __name__ == "__main__":
    import asyncio

    async def _quick_test():
        print("--- Gene Annotation live NCBI test ---")
        result = await get_gene_annotation(
            "GCF_000464555.1",
            user_question="Show me gene annotations for the tiger",
        )
        print("Tiger genes:", result)
        assert len(result["gene_list"]) > 0, "Expected at least one gene from NCBI"
        print("All tests passed ✅")

    asyncio.run(_quick_test())
