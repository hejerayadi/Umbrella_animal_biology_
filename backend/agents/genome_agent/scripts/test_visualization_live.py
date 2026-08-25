"""
Live visualization test — real NCBI data, real SVG output.

Calls the actual subagent pipeline:
    resolve_species  (NCBI taxonomy + assembly search)
    get_genome_metadata  (NCBI assembly stats)
    get_gene_annotation  (NCBI gene search + summaries)
    generate_visualization  (chromosome map + size comparison)

Writes SVGs to scripts/output/ and opens them in the browser.

Usage (from the repo root):
    python -m backend.agents.genome_agent.scripts.test_visualization_live

Optional env var:
    VIZ_SPECIES   species to query  (default: "Amur tiger")
    VIZ_SCOPE     chromosome_map | size_comparison | both  (default: both)

No NVIDIA_API_KEY needed for rendering — the LLM is only used by the
query router and explanation writer, both of which have deterministic
fallbacks. NCBI calls are unauthenticated.
"""

from __future__ import annotations

import asyncio
import os
import pathlib
import webbrowser

_OUT_DIR = pathlib.Path(__file__).parent / "output"
_OUT_DIR.mkdir(parents=True, exist_ok=True)

SPECIES  = os.getenv("VIZ_SPECIES", "Amur tiger")
SCOPE    = os.getenv("VIZ_SCOPE",   "both")          # chromosome_map | size_comparison | both


# ── helpers ───────────────────────────────────────────────────────────────────

def _save(name: str, svg_bytes: bytes) -> pathlib.Path:
    path = _OUT_DIR / name
    path.write_bytes(svg_bytes)
    print(f"  saved  → {path}  ({len(svg_bytes):,} bytes)")
    return path


def _html_report(title: str, sections: list[tuple[str, str]]) -> pathlib.Path:
    """Build a single HTML page.  sections = [(heading, content_html), ...]"""
    body = ""
    for heading, content in sections:
        body += f"<h2>{heading}</h2>\n{content}\n"
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{title}</title>
  <style>
    body {{ font-family: sans-serif; background: #fafafa; padding: 2em; max-width: 1000px; margin: 0 auto; }}
    h1   {{ color: #333; }}
    h2   {{ color: #555; margin-top: 2em; border-bottom: 1px solid #ddd; padding-bottom: .3em; }}
    pre  {{ background: #f0f0f0; padding: 1em; border-radius: 4px; overflow-x: auto; font-size: 13px; }}
    img  {{ max-width: 100%; border: 1px solid #ddd; border-radius: 4px; margin-top: .5em; }}
    .ok  {{ color: #2a7a2a; font-weight: bold; }}
    .err {{ color: #c0392b; font-weight: bold; }}
    .dim {{ color: #888; }}
  </style>
</head>
<body>
<h1>{title}</h1>
{body}
</body>
</html>"""
    out = _OUT_DIR / "live_report.html"
    out.write_text(html, encoding="utf-8")
    return out


def _pre(text: str) -> str:
    return f"<pre>{text}</pre>"


def _svg_img(path: pathlib.Path) -> str:
    return f'<img src="{path.name}" alt="{path.stem}" />'


# ── pipeline ──────────────────────────────────────────────────────────────────

async def run_live_test() -> None:
    from backend.agents.genome_agent.subagents.species_resolver import (
        resolve_species,
    )
    from backend.agents.genome_agent.subagents.genome_metadata import (
        get_genome_metadata,
    )
    from backend.agents.genome_agent.subagents.gene_annotation import (
        get_gene_annotation,
    )
    from backend.agents.genome_agent.subagents.visualization import (
        generate_visualization,
    )

    sections: list[tuple[str, str]] = []
    print(f"\n{'='*60}")
    print(f"  Live visualization test — species: {SPECIES!r}")
    print(f"{'='*60}\n")

    # ── Step 1: resolve species ───────────────────────────────────────────────
    print(f"[1/4] Resolving species: {SPECIES!r} via NCBI …")
    species = await resolve_species(SPECIES)

    assembly_id    = species.get("assembly_id")
    scientific     = species.get("scientific_name", "unknown")
    common         = species.get("common_name", SPECIES)

    status_tag = '<span class="ok">✓ resolved</span>' if assembly_id else '<span class="err">✗ not found</span>'
    sections.append(("1 · Species resolver", _pre(
        f"Query:           {SPECIES}\n"
        f"Scientific name: {scientific}\n"
        f"Common name:     {common}\n"
        f"Assembly ID:     {assembly_id or 'NOT FOUND'}\n"
        f"Confidence:      {species.get('confidence', 'n/a')}\n"
        f"Status:          {status_tag}"
    )))
    print(f"  assembly_id  = {assembly_id}")
    print(f"  scientific   = {scientific}")

    if not assembly_id:
        print("\n  ✗ Species not found — cannot continue.\n")
        report = _html_report(f"Live viz test — {SPECIES}", sections)
        webbrowser.open(report.as_uri())
        return

    # ── Step 2: genome metadata ───────────────────────────────────────────────
    print(f"\n[2/4] Fetching genome metadata for {assembly_id} …")
    metadata = await get_genome_metadata(assembly_id)

    genome_size_bp     = metadata.get("genome_size_bp")
    chromosome_count   = metadata.get("chromosome_count")
    assembly_level     = metadata.get("assembly_level")

    sections.append(("2 · Genome metadata", _pre(
        f"Assembly:          {assembly_id}\n"
        f"Assembly level:    {assembly_level}\n"
        f"Genome size:       {genome_size_bp:,} bp  ({genome_size_bp/1e9:.2f} Gb)"
            if genome_size_bp else
        f"Genome size:       not available\n"
        f"Chromosome count:  {chromosome_count or 'not available'}"
    )))
    print(f"  genome_size_bp     = {genome_size_bp}")
    print(f"  chromosome_count   = {chromosome_count}")
    print(f"  assembly_level     = {assembly_level}")

    # ── Step 3: gene annotation ───────────────────────────────────────────────
    print(f"\n[3/4] Fetching gene annotation for {assembly_id} …")
    annotation = await get_gene_annotation(
        assembly_id,
        user_question=f"Show gene map for {common}",
    )

    gene_table  = annotation.get("gene_table", [])
    gene_list   = annotation.get("gene_list",  [])

    rows_html = "".join(
        f"<tr><td>{g.get('gene_name','')}</td><td>{g.get('location','')}</td>"
        f"<td style='font-size:12px'>{g.get('function','')[:120]}</td></tr>"
        for g in gene_table[:20]
    )
    table_html = (
        f"<p>Showing first {min(len(gene_table),20)} of {len(gene_table)} genes "
        f"({len(gene_list)} in gene_list)</p>"
        f"<table border='1' cellpadding='4' style='border-collapse:collapse;font-size:13px'>"
        f"<tr><th>Gene</th><th>Location</th><th>Function</th></tr>"
        f"{rows_html}</table>"
    ) if gene_table else "<p class='err'>No genes returned from NCBI.</p>"

    sections.append(("3 · Gene annotation", table_html))
    print(f"  genes returned   = {len(gene_list)}")
    if gene_list:
        print(f"  first 10         = {gene_list[:10]}")

    # ── Step 4: generate visualizations ──────────────────────────────────────
    svg_sections: list[tuple[str, str]] = []

    if SCOPE in ("chromosome_map", "both"):
        print(f"\n[4a/4] Generating chromosome_map …")
        result = await generate_visualization(
            scope="chromosome_map",
            genome_size_bp=genome_size_bp,
            gene_table=gene_table,
            assembly_id=assembly_id,
            common_name=common,
            scientific_name=scientific,
            user_question=f"Show chromosome map for {common}",
        )
        status = result.get("status")
        chart  = result.get("chart_data")
        note   = result.get("note", "")
        print(f"  status = {status},  chart_data = {'bytes' if chart else None},  note = {note!r}")

        if isinstance(chart, bytes) and chart:
            p = _save(f"live_chromosome_map.svg", chart)
            svg_sections.append(("4a · Chromosome map (live NCBI)", _svg_img(p)))
        else:
            msg = note or f"status={status} — no SVG produced"
            svg_sections.append(("4a · Chromosome map (live NCBI)", f"<p class='err'>{msg}</p>"))

    if SCOPE in ("size_comparison", "both"):
        print(f"\n[4b/4] Generating size_comparison …")
        result = await generate_visualization(
            scope="size_comparison",
            genome_size_bp=genome_size_bp,
            gene_table=gene_table,
            assembly_id=assembly_id,
            common_name=common,
            scientific_name=scientific,
            user_question=f"Compare genome size of {common} to other species",
        )
        status      = result.get("status")
        chart       = result.get("chart_data")
        comparisons = result.get("comparisons", [])
        note        = result.get("note", "")
        print(f"  status = {status},  species compared = {len(comparisons)},  note = {note!r}")

        if isinstance(chart, bytes) and chart:
            p = _save("live_size_comparison.svg", chart)
            comp_rows = "".join(
                f"<tr><td>{c['common_name']}</td>"
                f"<td>{c.get('scientific_name','')}</td>"
                f"<td>{c['genome_size_bp']/1e9:.2f} Gb</td>"
                f"<td>{'<b>← queried</b>' if c.get('is_queried_species') else ''}</td></tr>"
                for c in comparisons
            )
            comp_table = (
                f"<table border='1' cellpadding='4' style='border-collapse:collapse;font-size:13px;margin-bottom:1em'>"
                f"<tr><th>Species</th><th>Scientific name</th><th>Genome size</th><th></th></tr>"
                f"{comp_rows}</table>"
            )
            svg_sections.append(("4b · Genome size comparison (live NCBI)", comp_table + _svg_img(p)))
        else:
            msg = note or f"status={status} — no SVG produced"
            svg_sections.append(("4b · Genome size comparison (live NCBI)", f"<p class='err'>{msg}</p>"))

    sections.extend(svg_sections)

    # ── Report ────────────────────────────────────────────────────────────────
    report = _html_report(f"Live viz test — {SPECIES}", sections)
    print(f"\n  report → {report}")
    webbrowser.open(report.as_uri())
    print("\n✅  Done.\n")


# ── entry point ───────────────────────────────────────────────────────────────

# Resolve species uses a thin wrapper that may or may not exist yet —
# check which public function the module exposes.
async def _resolve_species_compat(name: str) -> dict:
    """Call whichever top-level resolver the module exposes."""
    from backend.agents.genome_agent.subagents import species_resolver as _sr
    if hasattr(_sr, "resolve_species"):
        return await _sr.resolve_species(name)
    # Fallback: use the direct NCBI path without LLM
    from backend.agents.genome_agent.subagents.species_resolver import (
        _search_taxonomy_core,
        _search_assembly_by_taxid_core,
    )
    candidates = await _search_taxonomy_core(name)
    if not candidates:
        return {"assembly_id": None, "scientific_name": None, "common_name": name, "confidence": 0.0}
    best = candidates[0]
    assemblies = await _search_assembly_by_taxid_core(str(best["tax_id"]))
    if not assemblies:
        return {"assembly_id": None, "scientific_name": best.get("scientific_name"), "common_name": best.get("common_name", name), "confidence": 0.3}
    return {
        "assembly_id": assemblies[0]["assembly_id"],
        "scientific_name": best.get("scientific_name"),
        "common_name": best.get("common_name", name),
        "confidence": 0.8,
    }


# Monkey-patch so run_live_test uses the compat wrapper when needed
import backend.agents.genome_agent.subagents.species_resolver as _sr_mod
if not hasattr(_sr_mod, "resolve_species"):
    _sr_mod.resolve_species = _resolve_species_compat  # type: ignore[attr-defined]


if __name__ == "__main__":
    asyncio.run(run_live_test())
