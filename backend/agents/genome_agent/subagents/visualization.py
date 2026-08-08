"""
Visualization — mock subagent (Task 1)
Generates a chromosome map or size-comparison chart from genome and gene data.
Renders locally for chromosome_map / size_comparison.
Delegates (NEEDS_AGENT) for protein_structure requests.

size_comparison now renders a real SVG bar chart comparing the queried
species' genome size against every other species known to the mock
species/metadata DBs, instead of returning a hardcoded placeholder byte
string. chromosome_map and protein_structure are unchanged (still mocked).
"""

from .genome_metadata import get_all_genome_metadata
from .species_resolver import get_all_species

_BAR_HEIGHT = 28
_BAR_GAP = 16
_LEFT_MARGIN = 160
_RIGHT_MARGIN = 90
_TOP_MARGIN = 40
_CHART_WIDTH = 560
_CURRENT_COLOR = "#e8622c"   # highlighted species (the one the user asked about)
_OTHER_COLOR = "#4b7ba5"
_AXIS_COLOR = "#333333"


def _format_bp(genome_size_bp: int) -> str:
    """Human-readable genome size, e.g. 2489000000 -> '2.49 Gb'."""
    return f"{genome_size_bp / 1_000_000_000:.2f} Gb"


def _build_size_comparison_svg(rows: list[dict], current_assembly_id: str | None) -> str:
    """
    rows: list of {"common_name", "scientific_name", "assembly_id", "genome_size_bp"}
          sorted descending by genome_size_bp.
    """
    max_size = max(r["genome_size_bp"] for r in rows) or 1
    chart_height = _TOP_MARGIN + len(rows) * (_BAR_HEIGHT + _BAR_GAP) + _BAR_GAP
    svg_width = _LEFT_MARGIN + _CHART_WIDTH + _RIGHT_MARGIN

    bars = []
    for i, r in enumerate(rows):
        y = _TOP_MARGIN + i * (_BAR_HEIGHT + _BAR_GAP)
        bar_w = round((r["genome_size_bp"] / max_size) * _CHART_WIDTH, 1)
        is_current = current_assembly_id is not None and r["assembly_id"] == current_assembly_id
        color = _CURRENT_COLOR if is_current else _OTHER_COLOR
        label = r["common_name"] + (" (queried)" if is_current else "")

        bars.append(
            f'<text x="{_LEFT_MARGIN - 10}" y="{y + _BAR_HEIGHT * 0.65}" '
            f'text-anchor="end" font-size="14" font-family="sans-serif" '
            f'fill="{_AXIS_COLOR}">{label}</text>'
        )
        bars.append(
            f'<rect x="{_LEFT_MARGIN}" y="{y}" width="{bar_w}" height="{_BAR_HEIGHT}" '
            f'fill="{color}" rx="3"/>'
        )
        bars.append(
            f'<text x="{_LEFT_MARGIN + bar_w + 8}" y="{y + _BAR_HEIGHT * 0.65}" '
            f'font-size="13" font-family="sans-serif" fill="{_AXIS_COLOR}">'
            f'{_format_bp(r["genome_size_bp"])}</text>'
        )

    svg = (
        f'<svg viewBox="0 0 {svg_width} {chart_height}" '
        f'xmlns="http://www.w3.org/2000/svg" font-family="sans-serif">'
        f'<rect x="0" y="0" width="{svg_width}" height="{chart_height}" fill="#ffffff"/>'
        f'<text x="{_LEFT_MARGIN}" y="20" font-size="15" font-weight="bold" '
        f'fill="{_AXIS_COLOR}">Genome size comparison</text>'
        + "".join(bars)
        + "</svg>"
    )
    return svg


async def generate_visualization(
    scope: str,
    genome_size_bp: int = None,
    gene_table: list = None,
    assembly_id: str = None,
    common_name: str = None,
    scientific_name: str = None,
) -> dict:
    """
    Mock version of Visualization.
    Input: scope ("chromosome_map" | "size_comparison" | "protein_structure"),
           plus data to render. assembly_id/common_name/scientific_name identify
           the *queried* species so size_comparison can highlight it among peers.
    Output: dict matching VisualizationOutput shape (plus an extra
            "comparisons" field on size_comparison so the explanation writer
            can talk about real numbers instead of a generic placeholder note).
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
        all_species = get_all_species()
        all_metadata = get_all_genome_metadata()

        rows = []
        for sp in all_species:
            aid = sp.get("assembly_id")
            meta = all_metadata.get(aid)
            if not meta or meta.get("genome_size_bp") is None:
                continue
            rows.append(
                {
                    "assembly_id": aid,
                    "common_name": sp.get("common_name"),
                    "scientific_name": sp.get("scientific_name"),
                    "genome_size_bp": meta["genome_size_bp"],
                    "taxonomic_group": sp.get("taxonomic_group"),
                }
            )

        # Make sure the queried species itself is represented even if it
        # wasn't (for whatever reason) in the species DB snapshot above —
        # fall back to the genome_size_bp / assembly_id the caller passed in.
        if assembly_id and not any(r["assembly_id"] == assembly_id for r in rows) and genome_size_bp:
            rows.append(
                {
                    "assembly_id": assembly_id,
                    "common_name": common_name or assembly_id,
                    "scientific_name": scientific_name,
                    "genome_size_bp": genome_size_bp,
                    "taxonomic_group": None,
                }
            )

        if not rows:
            return {
                "status": "COMPLETED",
                "chart_data": None,
                "format": None,
                "note": "No genome size data available for any species to compare.",
            }

        # Prefer comparing against taxonomically related species (e.g. "other
        # cats" for a tiger query) so the chart is actually relevant to the
        # question, not just a dump of every species the mock knows about.
        # Falls back to the full set if the queried species has no group, or
        # if it's the only member of its group in the mock DB.
        queried_group = next(
            (r["taxonomic_group"] for r in rows if r["assembly_id"] == assembly_id),
            None,
        )
        filtered_note = None
        if queried_group:
            same_group_rows = [r for r in rows if r["taxonomic_group"] == queried_group]
            if len(same_group_rows) > 1:
                rows = same_group_rows
                filtered_note = f"Comparison limited to other {queried_group} species."

        rows.sort(key=lambda r: r["genome_size_bp"], reverse=True)

        svg = _build_size_comparison_svg(rows, current_assembly_id=assembly_id)

        comparisons = [
            {
                "common_name": r["common_name"],
                "scientific_name": r["scientific_name"],
                "genome_size_bp": r["genome_size_bp"],
                "is_queried_species": r["assembly_id"] == assembly_id,
            }
            for r in rows
        ]

        result = {
            "status": "COMPLETED",
            "chart_data": svg.encode("utf-8"),
            "format": "svg",
            "comparisons": comparisons,
        }
        if filtered_note:
            result["note"] = filtered_note
        return result

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
        assert result["target_agent"] is None

        result = await generate_visualization(
            "size_comparison",
            genome_size_bp=2489000000,
            assembly_id="GCF_000464555.1",
            common_name="Tiger",
            scientific_name="Panthera tigris",
        )
        print("Size comparison:", result)
        assert result["status"] == "COMPLETED"
        assert result["format"] == "svg"
        assert result["chart_data"].startswith(b"<svg")
        assert len(result["comparisons"]) == 3  # mouse, tiger, elephant
        assert any(c["is_queried_species"] for c in result["comparisons"])
        print("SVG length:", len(result["chart_data"]), "bytes")

        print("All tests passed ✅")

    asyncio.run(_quick_test())