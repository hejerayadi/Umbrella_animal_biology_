"""
Quick visual test for the genome agent visualization subagents.

Runs from the repository root — no server, no NCBI calls, no LLM needed.
Writes three SVG files to genome/backend/agents/genome_agent/scripts/output/
then opens them in your default browser automatically.

Usage (from repo root):
    python -m backend.agents.genome_agent.scripts.test_visualization

What it tests:
    1. chromosome_map  — basic gene strip
    2. chromosome_map  — with a highlighted gene
    3. size_comparison — bar chart with the queried species highlighted
    4. chromosome_map  — empty gene table edge case
"""
from __future__ import annotations

import base64
import os
import pathlib
import webbrowser

# ── bring the subagent renderers in directly (no HTTP, no agent running) ──
from backend.agents.genome_agent.subagents.visualization_render import (
    render_chromosome_map,
    render_size_comparison,
)

# Where to write the output files
_OUT_DIR = pathlib.Path(__file__).parent / "output"
_OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── sample data ──────────────────────────────────────────────────────────────

GENE_TABLE = [
    {"gene_name": "TP53",   "location": "chr17", "function": "tumor suppressor"},
    {"gene_name": "BRCA1",  "location": "chr17", "function": "DNA repair"},
    {"gene_name": "EGFR",   "location": "chr7",  "function": "growth factor receptor"},
    {"gene_name": "MYC",    "location": "chr8",  "function": "transcription factor"},
    {"gene_name": "KRAS",   "location": "chr12", "function": "GTPase signalling"},
    {"gene_name": "FGF5",   "location": "chr4",  "function": "hair-growth inhibitor"},
    {"gene_name": "UCP1",   "location": "chr4",  "function": "thermogenesis"},
    {"gene_name": "ADRB3",  "location": "chr8",  "function": "beta-3 adrenergic receptor"},
]

SIZE_ROWS = [
    {"assembly_id": "GCF_000001405.40", "common_name": "Human",        "scientific_name": "Homo sapiens",          "genome_size_bp": 3_100_000_000},
    {"assembly_id": "GCF_000001635.27", "common_name": "Mouse",        "scientific_name": "Mus musculus",          "genome_size_bp": 2_728_222_451},
    {"assembly_id": "GCF_000002315.6",  "common_name": "Chicken",      "scientific_name": "Gallus gallus",         "genome_size_bp": 1_065_000_000},
    {"assembly_id": "GCF_000003025.6",  "common_name": "Pig",          "scientific_name": "Sus scrofa",            "genome_size_bp": 2_501_912_388},
    {"assembly_id": "GCF_000001215.4",  "common_name": "Fruit fly",    "scientific_name": "Drosophila melanogaster","genome_size_bp":   143_000_000},
    {"assembly_id": "GCF_000002235.5",  "common_name": "Zebrafish",    "scientific_name": "Danio rerio",           "genome_size_bp": 1_371_719_383},
    {"assembly_id": "GCF_TIGER_001",    "common_name": "Amur tiger",   "scientific_name": "Panthera tigris altaica","genome_size_bp": 2_430_000_000},
]
# The "queried species" that gets the orange highlight in the bar chart
QUERIED_ASSEMBLY = "GCF_TIGER_001"


# ── helpers ───────────────────────────────────────────────────────────────────

def _save(name: str, svg_bytes: bytes) -> pathlib.Path:
    path = _OUT_DIR / name
    path.write_bytes(svg_bytes)
    print(f"  wrote {path} ({len(svg_bytes):,} bytes)")
    return path


def _html_page(title: str, *svg_paths: pathlib.Path) -> pathlib.Path:
    """Wrap one or more SVGs in a minimal HTML page for easier viewing."""
    imgs = "\n".join(
        f'<figure style="margin:2em 0">'
        f'<figcaption style="font-family:sans-serif;font-size:14px;color:#555">'
        f'{p.stem}'
        f'</figcaption>'
        f'<img src="{p.name}" style="max-width:900px;border:1px solid #ddd" />'
        f'</figure>'
        for p in svg_paths
    )
    html = f"""<!DOCTYPE html>
<html lang="en">
<head><meta charset="utf-8"><title>{title}</title></head>
<body style="background:#fafafa;padding:2em">
<h1 style="font-family:sans-serif">{title}</h1>
{imgs}
</body>
</html>"""
    out = _OUT_DIR / "index.html"
    out.write_text(html, encoding="utf-8")
    return out


# ── test cases ────────────────────────────────────────────────────────────────

def test_chromosome_map_basic() -> pathlib.Path:
    print("\n[1] chromosome_map — no highlight")
    svg = render_chromosome_map(GENE_TABLE, highlight_gene=None)
    assert isinstance(svg, bytes) and svg.startswith(b"<svg"), "expected SVG bytes"
    return _save("chromosome_map_basic.svg", svg)


def test_chromosome_map_highlight() -> pathlib.Path:
    print("\n[2] chromosome_map — FGF5 highlighted")
    svg = render_chromosome_map(GENE_TABLE, highlight_gene="FGF5")
    assert isinstance(svg, bytes) and svg.startswith(b"<svg"), "expected SVG bytes"
    assert b"#e8622c" in svg, "highlight colour not found — check render_chromosome_map"
    return _save("chromosome_map_highlight.svg", svg)


def test_size_comparison() -> pathlib.Path:
    print("\n[3] size_comparison — Amur tiger highlighted")
    svg = render_size_comparison(SIZE_ROWS, current_assembly_id=QUERIED_ASSEMBLY)
    assert isinstance(svg, bytes) and svg.startswith(b"<svg"), "expected SVG bytes"
    assert b"Amur tiger" in svg, "queried species not found in SVG"
    return _save("size_comparison.svg", svg)


def test_chromosome_map_empty() -> pathlib.Path:
    print("\n[4] chromosome_map — empty gene table (edge case)")
    svg = render_chromosome_map([], highlight_gene=None)
    assert isinstance(svg, bytes) and svg.startswith(b"<svg"), "expected SVG bytes"
    return _save("chromosome_map_empty.svg", svg)


def test_base64_roundtrip() -> None:
    """Verify the base64 encoding used in orchestrator_adapter round-trips correctly."""
    print("\n[5] base64 roundtrip (simulates orchestrator_adapter._visualization_summary)")
    svg = render_size_comparison(SIZE_ROWS, current_assembly_id=QUERIED_ASSEMBLY)
    encoded = base64.b64encode(svg).decode("ascii")
    decoded = base64.b64decode(encoded)
    assert decoded == svg, "base64 roundtrip failed"
    data_url = f"data:image/svg+xml;base64,{encoded}"
    print(f"  data URL length: {len(data_url):,} chars  ✓")


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print("Genome Agent — Visualization Subagent Test")
    print(f"Output directory: {_OUT_DIR}")
    print("=" * 60)

    p1 = test_chromosome_map_basic()
    p2 = test_chromosome_map_highlight()
    p3 = test_size_comparison()
    p4 = test_chromosome_map_empty()
    test_base64_roundtrip()

    # Bundle into a single HTML page and open it
    html = _html_page(
        "Genome Agent — Visualization Test Output",
        p1, p2, p3, p4,
    )
    print(f"\n  wrote {html}")
    print("\n✅  All assertions passed.")

    # Open in default browser
    webbrowser.open(html.as_uri())
    print("  Opened in browser.")


if __name__ == "__main__":
    main()
