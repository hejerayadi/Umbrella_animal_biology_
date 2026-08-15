"""
Visualization — LLM-guided, grounded visualization generation.

Generates chromosome maps and genome size comparisons with:
- LLM selecting sensible reference species for comparison (size_comparison)
- Pure SVG rendering (no side effects, deterministic)
- Grounding verification (all genome sizes from tool results)
- Deterministic delegation for protein_structure

GROUNDING RULE: Every fact (genome_size_bp, assembly_id) in the final answer
must come from a tool result in the same execution. No invented data.

Renders locally for chromosome_map / size_comparison.
Delegates (NEEDS_AGENT) for protein_structure requests.
"""

from __future__ import annotations

import asyncio
import logging

from ..workflows.visualization_resolver import (
    resolve_visualization_references,
    resolve_visualization_references_fallback,
)
from .genome_metadata import get_genome_metadata
from .species_resolver import resolve_species
from .visualization_render import render_chromosome_map, render_size_comparison

logger = logging.getLogger(__name__)


async def resolve_reference_species(
    candidate_names: list[str],
) -> list[dict]:
    """
    Resolve candidate reference species into grounded assembly data.
    
    Delegates to existing Species Resolver + Genome Metadata agents
    (colleagues' implementations). Any species that fails to resolve
    is silently dropped — a partial comparison is still useful.
    
    Args:
        candidate_names: List of species common names (e.g., ["human", "mouse"])
    
    Returns:
        List of dicts with keys:
        - assembly_id (str)
        - common_name (str)
        - scientific_name (str)
        - genome_size_bp (int)
        
        Failed candidates are dropped (not included in result).
    """
    if not candidate_names:
        return []
    
    # Step 1: Resolve each species name to assembly ID
    species_results = await asyncio.gather(
        *(resolve_species(name) for name in candidate_names),
        return_exceptions=True,
    )
    
    # Step 2: Collect species that resolved successfully
    resolved_species = []
    for name, result in zip(candidate_names, species_results):
        if isinstance(result, Exception):
            logger.warning("Failed to resolve species %r: %s", name, result)
            continue
        if not result.get("assembly_id"):
            logger.warning("Species %r resolved to None assembly_id", name)
            continue
        resolved_species.append(result)
    
    if not resolved_species:
        return []
    
    # Step 3: Fetch genome metadata for each resolved species
    metadata_results = await asyncio.gather(
        *(get_genome_metadata(sp["assembly_id"]) for sp in resolved_species),
        return_exceptions=True,
    )
    
    # Step 4: Combine species info + metadata, dropping failures
    grounded_rows = []
    for sp, meta in zip(resolved_species, metadata_results):
        if isinstance(meta, Exception):
            logger.warning("Failed to fetch metadata for %r: %s", sp.get("assembly_id"), meta)
            continue
        if meta.get("genome_size_bp") is None:
            logger.warning("No genome_size_bp for assembly %r", sp.get("assembly_id"))
            continue
        
        # Grounding: all fields come from tool results
        grounded_rows.append(
            {
                "assembly_id": sp["assembly_id"],
                "common_name": sp.get("common_name", "Unknown"),
                "scientific_name": sp.get("scientific_name"),
                "genome_size_bp": meta["genome_size_bp"],
            }
        )
    
    return grounded_rows


async def generate_visualization(
    scope: str,
    genome_size_bp: int | None = None,
    gene_table: list | None = None,
    assembly_id: str | None = None,
    common_name: str | None = None,
    scientific_name: str | None = None,
    user_question: str | None = None,
) -> dict:
    """
    Generate a visualization based on the requested scope.
    
    Args:
        scope: "chromosome_map" | "size_comparison" | "protein_structure"
        genome_size_bp: Queried species genome size (for size_comparison)
        gene_table: List of genes (for chromosome_map)
        assembly_id: Queried species assembly ID (for highlighting in comparisons)
        common_name: Common name of queried species
        scientific_name: Scientific name of queried species
        user_question: Original user question (for context)
    
    Returns:
        dict matching VisualizationOutput:
        {
            "status": "COMPLETED" | "NEEDS_AGENT" | "FAILED",
            "chart_data": bytes | None,
            "format": str | None,
            "target_agent": str | None,
            "prompt_to_target_agent": str | None,
            "note": str | None,
            "comparisons": list[dict] | None,
        }
    """
    
    if scope == "protein_structure":
        # Protein structure is deterministic: always delegate, never invoke LLM
        handoff_prompt = f"Render a 3D protein structure visualization"
        if user_question:
            handoff_prompt += f" for: {user_question}"
        if assembly_id:
            handoff_prompt += f" (Assembly: {assembly_id})"
        
        return {
            "status": "NEEDS_AGENT",
            "target_agent": None,
            "prompt_to_target_agent": handoff_prompt,
            "chart_data": None,
            "format": None,
        }
    
    if scope == "chromosome_map":
        # Chromosome map: use pure rendering, optionally highlight named gene
        if not gene_table:
            return {
                "status": "COMPLETED",
                "chart_data": None,
                "format": None,
                "note": "No gene data available to render a chromosome map.",
            }
        
        # Try to extract a gene name from the question to highlight
        highlight_gene = None
        if user_question:
            # Simple heuristic: look for capitalized words that might be gene names
            # In a real system, the LLM could extract this more intelligently
            words = user_question.split()
            for word in words:
                if len(word) <= 10 and word[0].isupper():
                    # Check if this gene name exists in our data
                    gene_names = [g.get("gene_name", "") for g in gene_table]
                    if word in gene_names:
                        highlight_gene = word
                        break
        
        chart_data = render_chromosome_map(gene_table, highlight_gene=highlight_gene)
        
        return {
            "status": "COMPLETED",
            "chart_data": chart_data,
            "format": "svg",
        }
    
    if scope == "size_comparison":
        # Size comparison: LLM selects reference species, then render
        
        # Step 1: Ask LLM for reference species candidates
        strategy = resolve_visualization_references(
            user_question or common_name or "unknown",
            current_common_name=common_name,
        )
        if strategy is None:
            strategy = resolve_visualization_references_fallback(
                common_name or "unknown"
            )
        
        logger.info(
            "[visualization] size_comparison strategy: candidates=%r, reasoning=%r",
            strategy.reference_species,
            strategy.reasoning,
        )
        
        # Step 2: Start with the queried species
        rows = []
        if assembly_id and genome_size_bp:
            rows.append(
                {
                    "assembly_id": assembly_id,
                    "common_name": common_name or assembly_id,
                    "scientific_name": scientific_name,
                    "genome_size_bp": genome_size_bp,
                }
            )
        
        # Step 3: Resolve LLM's candidate reference species
        # (drops any that fail to resolve or lack genome size)
        try:
            resolved_refs = await resolve_reference_species(strategy.reference_species)
            # Only add references that aren't already the queried species
            for ref in resolved_refs:
                if assembly_id and ref["assembly_id"] == assembly_id:
                    continue  # Already have the queried species
                rows.append(ref)
        except Exception as exc:
            logger.error("resolve_reference_species failed: %s", exc)
            # Continue with just the queried species if reference resolution fails
        
        if not rows:
            return {
                "status": "COMPLETED",
                "chart_data": None,
                "format": None,
                "note": "No genome size data available for any species to compare.",
            }
        
        # Step 4: Verify grounding (all genome sizes came from NCBI)
        for row in rows:
            if row.get("genome_size_bp") is None:
                logger.error(
                    "GROUNDING FAILURE: genome_size_bp is None for %r. "
                    "This value must come from get_genome_metadata().",
                    row.get("assembly_id"),
                )
                # Strict rejection: if grounding failed, return COMPLETED but with no data
                return {
                    "status": "COMPLETED",
                    "chart_data": None,
                    "format": None,
                    "note": "Grounding verification failed: some genome sizes not from NCBI.",
                }
        
        # Step 5: Sort and render
        rows.sort(key=lambda r: r["genome_size_bp"], reverse=True)
        chart_data = render_size_comparison(rows, current_assembly_id=assembly_id)
        
        comparisons = [
            {
                "common_name": r["common_name"],
                "scientific_name": r["scientific_name"],
                "genome_size_bp": r["genome_size_bp"],
                "is_queried_species": r["assembly_id"] == assembly_id if assembly_id else False,
            }
            for r in rows
        ]
        
        return {
            "status": "COMPLETED",
            "chart_data": chart_data,
            "format": "svg",
            "comparisons": comparisons,
        }
    
    # Unknown scope
    logger.warning("Unknown visualization scope: %r", scope)
    return {
        "status": "FAILED",
        "chart_data": None,
        "format": None,
    }


if __name__ == "__main__":
    import asyncio

    async def _quick_test():
        print("--- Visualization live NCBI test ---")
        
        result = await generate_visualization(
            "chromosome_map",
            genome_size_bp=2728222451,
            gene_table=[{"gene_name": "Trp53", "location": "chr17", "function": "tumor suppressor"}],
            user_question="Show chromosome map with Trp53 highlighted",
        )
        print("Chromosome map:", result)
        assert result["status"] == "COMPLETED"
        assert result["format"] == "svg"

        result = await generate_visualization(
            "chromosome_map",
            gene_table=[],
            user_question="Show map",
        )
        print("Empty gene table:", result)
        assert result["chart_data"] is None

        result = await generate_visualization("protein_structure")
        print("Protein structure:", result)
        assert result["status"] == "NEEDS_AGENT"

        print("All tests passed ✅")

    asyncio.run(_quick_test())