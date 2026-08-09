"""Folium-backed map renderer for the Biodiversity Agent.

Given a species name and a list of ``(lat, lon)`` observations, produce
an interactive Leaflet HTML map (via folium) and save it under
``outputs/maps/{species_slug}.html``. The dashboard and future frontend
render this HTML directly - no image generation, no map tile caching,
just a static file the browser opens as-is.

The renderer degrades gracefully:
- If ``folium`` is not installed, it returns a placeholder ``file://``
  URL so the mock's contract stays the same.
- If the coordinates list is empty, it raises ``ValueError`` - callers
  should have filtered that case out already.

Everything lives in this file - swapping folium for another engine
(mapbox, plotly, deck.gl) only touches ``render_point_map``.
"""

from __future__ import annotations

from pathlib import Path


# Where rendered maps land. Kept under the agent folder so nothing
# leaks outside ``biodiversity_agent/``.
_OUTPUT_DIR = Path(__file__).resolve().parents[2] / "outputs" / "maps"


def output_dir() -> Path:
    """Ensure the maps directory exists and return it."""

    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return _OUTPUT_DIR


def render_point_map(
    species_name: str,
    coordinates: list[tuple[float, float]],
    output_dir_override: Path | None = None,
) -> str:
    """Render a point map and return the ``file://`` URL of the saved HTML.

    Parameters
    ----------
    species_name
        Used both for the popup labels and the output filename.
    coordinates
        List of ``(lat, lon)`` pairs. Must be non-empty.
    output_dir_override
        For tests - lets the caller point at a tmp directory.
    """

    if not coordinates:
        raise ValueError("render_point_map requires at least one coordinate.")

    target_dir = output_dir_override or output_dir()
    target_dir.mkdir(parents=True, exist_ok=True)

    slug = species_name.lower().replace(" ", "_")
    out_path = target_dir / f"{slug}.html"

    try:
        import folium  # type: ignore
    except ImportError:
        # No folium installed - fall back to a placeholder URL so the
        # rest of the pipeline keeps working. Prints once, not per call.
        return f"file://{out_path.as_posix()}#folium-missing"

    # Centre the map on the mean of the observations.
    mean_lat = sum(lat for lat, _ in coordinates) / len(coordinates)
    mean_lon = sum(lon for _, lon in coordinates) / len(coordinates)

    fmap = folium.Map(
        location=[mean_lat, mean_lon],
        zoom_start=2 if len(coordinates) > 3 else 4,
        tiles="OpenStreetMap",
    )

    for lat, lon in coordinates:
        folium.CircleMarker(
            location=[lat, lon],
            radius=6,
            popup=folium.Popup(
                f"<b>{species_name}</b><br>({lat:.3f}, {lon:.3f})",
                max_width=250,
            ),
            color="#c8102e",
            fill=True,
            fill_color="#c8102e",
            fill_opacity=0.75,
            weight=1,
        ).add_to(fmap)

    # Add a small legend explaining the source.
    legend_html = f"""
    <div style="position: fixed; bottom: 20px; left: 20px; z-index: 9999;
                background: white; padding: 8px 12px; border: 1px solid #ccc;
                border-radius: 4px; font-family: sans-serif; font-size: 12px;">
      <b>{species_name}</b><br>
      {len(coordinates)} observations (mock)<br>
      <span style="color: #666;">Sprint 2 — Biodiversity Agent</span>
    </div>
    """
    fmap.get_root().html.add_child(folium.Element(legend_html))

    fmap.save(str(out_path))
    return f"file://{out_path.as_posix()}"
