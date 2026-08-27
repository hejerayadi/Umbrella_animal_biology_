"""
Visualization Rendering — pure SVG generation functions.

These are deterministic, side-effect-free functions that render visualizations.
No NCBI calls, no LLM calls, no network access. Same input produces same output.

This keeps rendering logic separate from LLM reasoning and NCBI data fetching.
"""

from __future__ import annotations


def render_chromosome_map(
    gene_table: list[dict],
    highlight_gene: str | None = None,
) -> bytes:
    """
    Render a chromosome map/gene visualization from gene data.
    
    Args:
        gene_table: List of dicts with keys: gene_name, location, function
        highlight_gene: Optional gene name to highlight/emphasize
    
    Returns:
        SVG content as bytes
    
    Pure function: no network, no LLM, no state.
    """
    if not gene_table:
        # Empty visualization
        return b'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 100"><text x="50" y="50">No gene data available</text></svg>'
    
    # Build a simple SVG with gene boxes
    width = 800
    height = 100 + len(gene_table) * 30
    
    svg_parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}">',
        '<rect x="0" y="0" width="' + str(width) + '" height="' + str(height) + '" fill="#f9f9f9"/>',
        '<text x="20" y="25" font-size="16" font-weight="bold" fill="#333">Chromosome Map</text>',
    ]
    
    for i, gene in enumerate(gene_table):
        y = 60 + i * 30
        gene_name = gene.get("gene_name", "Unknown")
        location = gene.get("location", "N/A")
        
        # Highlight if this is the requested gene
        is_highlight = highlight_gene and gene_name.lower() == highlight_gene.lower()
        color = "#e8622c" if is_highlight else "#4b7ba5"
        
        # Draw gene box
        svg_parts.append(
            f'<rect x="20" y="{y-15}" width="150" height="20" fill="{color}" rx="3"/>'
        )
        # Draw gene name
        svg_parts.append(
            f'<text x="30" y="{y-2}" font-size="12" fill="white">{gene_name}</text>'
        )
        # Draw location
        svg_parts.append(
            f'<text x="180" y="{y-2}" font-size="11" fill="#666">{location}</text>'
        )
    
    svg_parts.append('</svg>')
    
    return '\n'.join(svg_parts).encode('utf-8')


def render_size_comparison(
    rows: list[dict],
    current_assembly_id: str | None = None,
) -> bytes:
    """
    Render a genome size comparison chart (SVG bar chart).
    
    Args:
        rows: List of dicts with keys: common_name, assembly_id, genome_size_bp
        current_assembly_id: Assembly ID of the queried species (to highlight)
    
    Returns:
        SVG content as bytes
    
    Pure function: no network, no LLM, no state.
    
    Expected row structure:
        {"common_name": "tiger", "assembly_id": "GCF_xxx", "genome_size_bp": 2.7e9}
    """
    if not rows:
        return b'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 100"><text x="50" y="50">No comparison data</text></svg>'
    
    # Sort descending by genome size (largest first)
    rows_sorted = sorted(rows, key=lambda r: r.get("genome_size_bp", 0), reverse=True)
    
    # Constants for layout
    bar_height = 28
    bar_gap = 16
    left_margin = 160
    right_margin = 90
    top_margin = 40
    chart_width = 560
    current_color = "#e8622c"   # Highlighted species
    other_color = "#4b7ba5"
    axis_color = "#333333"
    
    chart_height = top_margin + len(rows_sorted) * (bar_height + bar_gap) + bar_gap
    svg_width = left_margin + chart_width + right_margin
    
    def format_bp(genome_size_bp: int) -> str:
        """Human-readable genome size."""
        if genome_size_bp is None:
            return "N/A"
        return f"{genome_size_bp / 1_000_000_000:.2f} Gb"
    
    svg_parts = [
        f'<svg viewBox="0 0 {svg_width} {chart_height}" xmlns="http://www.w3.org/2000/svg" font-family="sans-serif">',
        f'<rect x="0" y="0" width="{svg_width}" height="{chart_height}" fill="#ffffff"/>',
        f'<text x="{left_margin}" y="20" font-size="15" font-weight="bold" fill="{axis_color}">Genome size comparison</text>',
    ]
    
    max_size = max((r.get("genome_size_bp") or 1) for r in rows_sorted) or 1
    
    for i, row in enumerate(rows_sorted):
        y = top_margin + i * (bar_height + bar_gap)
        genome_size = row.get("genome_size_bp") or 0
        common_name = row.get("common_name", "Unknown")
        assembly_id = row.get("assembly_id", "")
        
        # Determine if this is the queried species
        is_current = current_assembly_id is not None and assembly_id == current_assembly_id
        color = current_color if is_current else other_color
        label = common_name + (" (queried)" if is_current else "")
        
        # Calculate bar width proportional to genome size
        bar_w = round((genome_size / max_size) * chart_width, 1) if max_size > 0 else 0
        
        # Draw species label (right-aligned, left of chart)
        svg_parts.append(
            f'<text x="{left_margin - 10}" y="{y + bar_height * 0.65}" '
            f'text-anchor="end" font-size="14" font-family="sans-serif" '
            f'fill="{axis_color}">{label}</text>'
        )
        
        # Draw bar
        svg_parts.append(
            f'<rect x="{left_margin}" y="{y}" width="{bar_w}" height="{bar_height}" '
            f'fill="{color}" rx="3"/>'
        )
        
        # Draw genome size label (right of bar)
        svg_parts.append(
            f'<text x="{left_margin + bar_w + 8}" y="{y + bar_height * 0.65}" '
            f'font-size="13" font-family="sans-serif" fill="{axis_color}">'
            f'{format_bp(genome_size)}</text>'
        )
    
    svg_parts.append('</svg>')
    
    return '\n'.join(svg_parts).encode('utf-8')
