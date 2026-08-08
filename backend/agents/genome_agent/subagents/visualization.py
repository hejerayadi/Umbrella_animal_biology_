"""
Visualization — mock subagent (Task 1)
Generates a chromosome map or size-comparison chart from genome and gene data.
Renders locally for chromosome_map / size_comparison.
Delegates (NEEDS_AGENT) for protein_structure requests.
Currently returns hardcoded fake data — no real rendering yet.
"""


async def generate_visualization(scope: str, genome_size_bp: int = None, gene_table: list = None) -> dict:
    """
    Mock version of Visualization.
    Input: scope ("chromosome_map" | "size_comparison" | "protein_structure"), plus data to render
    Output: dict matching VisualizationOutput shape
    """

    if scope == "protein_structure":
        return {
            "status": "NEEDS_AGENT",
            "target_agent": None,
            "prompt_to_target_agent": "Render an interactive, labeled 3D structure for the requested gene.",
            "chart_data": None,
            "format": None,
        }

    if scope == "chromosome_map":
        if not gene_table:
            return {
                "status": "COMPLETED",
                "chart_data": None,
                "format": None,
                "note": "No gene data available to render a chromosome map.",
            }
        return {
            "status": "COMPLETED",
            "chart_data": b"<fake_svg_chromosome_map>",
            "format": "svg",
        }

    if scope == "size_comparison":
        return {
            "status": "COMPLETED",
            "chart_data": b"<fake_svg_size_comparison>",
            "format": "svg",
        }

    # Unknown scope
    return {
        "status": "FAILED",
        "chart_data": None,
        "format": None,
    }


if __name__ == "__main__":
    import asyncio

    async def _quick_test():
        result = await generate_visualization(
            "chromosome_map",
            genome_size_bp=2728222451,
            gene_table=[{"gene_name": "Trp53"}],
        )
        print("Chromosome map:", result)
        assert result["status"] == "COMPLETED"
        assert result["format"] == "svg"

        result = await generate_visualization("chromosome_map", gene_table=[])
        print("Empty gene table:", result)
        assert result["chart_data"] is None

        result = await generate_visualization("protein_structure")
        print("Protein structure:", result)
        assert result["status"] == "NEEDS_AGENT"
        assert result["target_agent"] == "protein_structure_visualization_agent"

        print("All tests passed ✅")

    asyncio.run(_quick_test())
