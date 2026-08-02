"""
Genome Agent Orchestrator (Task 1 — Sprint 1)
Coordinates the 4 mock subagents in the required order and parallelism.

Call order:
  1. resolve_species()                             — sequential, gating step
  2. get_genome_metadata() + get_gene_annotation() — parallel (asyncio.gather)
  3. generate_visualization()                      — sequential, uses step 2 outputs

Fallback rules:
  - assembly_id is None → stop early, return species_not_found error
  - metadata or annotation fails/empty → include whatever succeeded, keep going
  - visualization status == "NEEDS_AGENT" → bubble up as-is, never retry
"""

import asyncio

from .subagents.species_resolver import resolve_species
from .subagents.genome_metadata import get_genome_metadata
from .subagents.gene_annotation import get_gene_annotation
from .subagents.visualization import generate_visualization


class GenomeAgentOrchestrator:
    def __init__(self):
        pass

    async def run(self, species_name: str, visualization_scope: str = "chromosome_map") -> dict:
        """
        Main entry point for the Genome Agent.

        Args:
            species_name:        Common or scientific name of the target species.
            visualization_scope: One of "chromosome_map", "size_comparison",
                                 or "protein_structure". Defaults to "chromosome_map".

        Returns:
            A single merged output dict with keys:
              - species      : resolved species info (always present)
              - metadata     : genome metadata (None if unavailable)
              - annotation   : gene annotation  (None if unavailable)
              - visualization: visualization result, may carry status="NEEDS_AGENT"
              - errors       : list of non-fatal error messages (may be empty)
        """

        output = {
            "species": None,
            "metadata": None,
            "annotation": None,
            "visualization": None,
            "errors": [],
        }

        # ------------------------------------------------------------------
        # Step 1 — Resolve species (sequential, gating)
        # ------------------------------------------------------------------
        try:
            species = await resolve_species(species_name)
        except Exception as exc:
            output["errors"].append(f"species_resolver raised an exception: {exc}")
            return output

        output["species"] = species

        if species.get("assembly_id") is None:
            # Hard stop — everything downstream needs an assembly_id
            output["errors"].append(
                f"Species '{species_name}' could not be resolved to a genome assembly. "
                "No further data can be retrieved."
            )
            return output

        assembly_id = species["assembly_id"]

        # ------------------------------------------------------------------
        # Step 2 — Metadata + Annotation in parallel
        # ------------------------------------------------------------------
        metadata_result, annotation_result = await asyncio.gather(
            get_genome_metadata(assembly_id),
            get_gene_annotation(assembly_id),
            return_exceptions=True,  # prevents one failure from cancelling the other
        )

        # Unpack metadata
        if isinstance(metadata_result, Exception):
            output["errors"].append(f"get_genome_metadata raised an exception: {metadata_result}")
            metadata = None
        elif metadata_result.get("genome_size_bp") is None:
            output["errors"].append(
                f"Genome metadata returned empty for assembly '{assembly_id}'."
            )
            metadata = None
        else:
            metadata = metadata_result

        output["metadata"] = metadata

        # Unpack annotation
        if isinstance(annotation_result, Exception):
            output["errors"].append(f"get_gene_annotation raised an exception: {annotation_result}")
            annotation = None
        elif not annotation_result.get("gene_list"):
            # Empty gene list is not necessarily an error — log it softly
            output["errors"].append(
                f"Gene annotation returned no genes for assembly '{assembly_id}'."
            )
            annotation = annotation_result  # keep it (has empty lists, not None)
        else:
            annotation = annotation_result

        output["annotation"] = annotation

        # ------------------------------------------------------------------
        # Step 3 — Visualization (sequential, uses step 2 outputs)
        # ------------------------------------------------------------------
        genome_size = metadata["genome_size_bp"] if metadata else None
        gene_table = annotation["gene_table"] if annotation else None

        try:
            viz = await generate_visualization(
                scope=visualization_scope,
                genome_size_bp=genome_size,
                gene_table=gene_table,
            )
        except Exception as exc:
            output["errors"].append(f"generate_visualization raised an exception: {exc}")
            viz = None

        # If the visualization needs another agent, bubble it up unchanged — never resolve here
        output["visualization"] = viz

        return output


# --------------------------------------------------------------------------
# Quick sanity-check — run with: python -m backend.agents.genome_agent.orchestrator
# --------------------------------------------------------------------------
if __name__ == "__main__":

    async def _run_tests():
        orch = GenomeAgentOrchestrator()
        sep = "-" * 60

        # Test 1: Known species — happy path
        print(sep)
        print("TEST 1 — Tiger, chromosome_map (happy path)")
        print(sep)
        result = await orch.run("tiger", visualization_scope="chromosome_map")
        print(f"  Species   : {result['species']}")
        print(f"  Metadata  : {result['metadata']}")
        print(f"  Annotation: {result['annotation']}")
        print(f"  Viz       : {result['visualization']}")
        print(f"  Errors    : {result['errors']}")
        assert result["species"]["assembly_id"] == "GCF_000464555.1"
        assert result["metadata"]["chromosome_count"] == 38
        assert result["annotation"]["gene_list"] == ["Mc1r"]
        assert result["visualization"]["status"] == "COMPLETED"
        assert not result["errors"]
        print("  ✅ PASSED\n")

        # Test 2: Unknown species — early stop
        print(sep)
        print("TEST 2 — Unknown species 'dragon' (early-stop fallback)")
        print(sep)
        result = await orch.run("dragon")
        print(f"  Species   : {result['species']}")
        print(f"  Metadata  : {result['metadata']}")
        print(f"  Annotation: {result['annotation']}")
        print(f"  Viz       : {result['visualization']}")
        print(f"  Errors    : {result['errors']}")
        assert result["species"]["assembly_id"] is None
        assert result["metadata"] is None
        assert result["annotation"] is None
        assert result["visualization"] is None
        assert len(result["errors"]) == 1
        assert "could not be resolved" in result["errors"][0]
        print("  ✅ PASSED\n")

        # Test 3: protein_structure — NEEDS_AGENT must bubble up
        print(sep)
        print("TEST 3 — House mouse, protein_structure scope (NEEDS_AGENT bubble-up)")
        print(sep)
        result = await orch.run("house mouse", visualization_scope="protein_structure")
        print(f"  Species   : {result['species']}")
        print(f"  Metadata  : {result['metadata']}")
        print(f"  Annotation: {result['annotation']}")
        print(f"  Viz       : {result['visualization']}")
        print(f"  Errors    : {result['errors']}")
        assert result["species"]["assembly_id"] == "GCF_000001635.27"
        assert result["visualization"]["status"] == "NEEDS_AGENT"
        assert result["visualization"]["target_agent"] == "protein_structure_visualization_agent"
        assert "prompt_to_target_agent" in result["visualization"]
        print("  ✅ PASSED\n")

        print(sep)
        print("All orchestrator tests passed ✅")
        print(sep)

    asyncio.run(_run_tests())
