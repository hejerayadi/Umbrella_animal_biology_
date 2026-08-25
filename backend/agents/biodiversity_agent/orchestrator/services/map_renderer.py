"""Folium-backed map renderer for the four Biodiversity workers.

This module is the shared map service used by every worker in the
Biodiversity Agent domain. It exposes one high-level function per
visualization type - four in total, one per skill worker:

- ``render_point_map``    — Species Distribution (Aziz).
- ``render_habitat_map``  — Habitat Visualization (Ouissale).
- ``render_heatmap``      — Biodiversity Hotspots (Ouissale).
- ``render_route_map``    — Migration Analysis (Mariem).

All four functions return a ``file://`` URL pointing to a self-contained
Leaflet HTML file under ``outputs/maps/{slug}.html``. The HTML embeds
its own JS/CSS and needs no server to display — the dashboard renders
it inline via ``st.components.v1.html``.

Design principles
-----------------
* **Same contract everywhere.** Every ``render_*`` returns a URL string;
  every worker mock can be swapped for a real worker without changing
  the orchestrator.
* **One base map, one style.** ``_create_base_map`` sets the tile,
  bounds, and layer-control scaffolding used by all four renderers so
  the aesthetics stay consistent across skills.
* **Graceful degradation.** If ``folium`` is not installed the module
  returns a placeholder ``file://`` URL — the worker's contract holds
  either way, and the test suite runs without map deps.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable


# Where rendered maps land.
_OUTPUT_DIR = Path(__file__).resolve().parents[2] / "outputs" / "maps"


def output_dir() -> Path:
    """Ensure the maps directory exists and return it."""

    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return _OUTPUT_DIR


# ---------- shared helpers ----------


_TILE_LIGHT = "cartodbpositron"           # clean grey/white basemap
_TILE_DARK = "cartodbdark_matter"         # dark alt basemap for LayerControl


def _folium_or_none():
    """Return the folium module if available, else ``None``.

    All renderers call this once at the top and fall back to a
    placeholder URL when it returns ``None``.
    """

    try:
        import folium  # type: ignore
        return folium
    except ImportError:
        return None


def _slug(name: str) -> str:
    return name.lower().replace(" ", "_").replace("/", "_")


def _placeholder_url(slug: str) -> str:
    # Uses .as_uri() so the returned string is a valid RFC 8089 file URI on
    # every OS (Windows produces file:///C:/... with three slashes, Linux
    # produces file:///path with just one). The dashboard parses it back
    # into a Path via urllib.parse.urlparse.
    return (_OUTPUT_DIR / f"{slug}.html").resolve().as_uri() + "#folium-missing"


def _bounds(coordinates: Iterable[tuple[float, float]]) -> tuple[list[float], list[float]]:
    """Return ``(sw, ne)`` bounds for ``fit_bounds()``. Assumes ``coordinates``
    is non-empty."""

    lats = [c[0] for c in coordinates]
    lons = [c[1] for c in coordinates]
    return [min(lats), min(lons)], [max(lats), max(lons)]


def _confidence_color(confidence: float | None) -> str:
    """Green/orange/yellow gradient. Used by the point marker layer to
    encode confidence visually without a separate legend."""

    if confidence is None:
        return "#2C5F2D"   # default forest green
    if confidence >= 0.80:
        return "#2E7D32"   # strong green
    if confidence >= 0.50:
        return "#EF6C00"   # orange
    return "#F9A825"       # yellow (low confidence)


def _create_base_map(folium, center: tuple[float, float], zoom_start: int = 3):
    """Return a folium.Map preconfigured with our shared style.

    * CartoDB Positron is the *default* base tile (light, clean). It is
      loaded through ``folium.Map(tiles=...)`` so LayerControl cannot
      accidentally switch to a heavier basemap on first render.
    * CartoDB Dark Matter is offered as a togglable alternative.
    * A ``Fullscreen`` button (top-left) and a ``MiniMap`` overview
      (bottom-right) are added to every map so the demo feels polished.
    """

    from folium.plugins import Fullscreen, MiniMap

    fmap = folium.Map(
        location=list(center),
        zoom_start=zoom_start,
        tiles=None,           # add named tiles explicitly for clean LayerControl labels
        control_scale=True,
        zoom_control=True,
    )
    # Human-readable names in the LayerControl radio group. First-added
    # layer with ``show=True`` becomes the default visible basemap.
    folium.TileLayer(
        _TILE_LIGHT, name="Light basemap", control=True, show=True, overlay=False,
        attr="© OpenStreetMap contributors · © CARTO",
    ).add_to(fmap)
    folium.TileLayer(
        _TILE_DARK, name="Dark basemap", control=True, show=False, overlay=False,
        attr="© OpenStreetMap contributors · © CARTO",
    ).add_to(fmap)

    # Polish plugins.
    Fullscreen(position="topleft", title="Fullscreen", title_cancel="Exit").add_to(fmap)
    MiniMap(tile_layer=_TILE_LIGHT, position="bottomright", toggle_display=True,
            width=140, height=100, zoom_level_offset=-5).add_to(fmap)

    # Shared CSS for numbered pulsing markers and permanent tooltips.
    _inject_shared_css(fmap, folium)
    return fmap


def _inject_shared_css(fmap, folium) -> None:
    """Inject the pulse keyframes and tooltip styling into the map <head>
    exactly once. Every ``DivIcon`` marker then references these classes
    instead of shipping its own inline animation."""

    css = """
    <style>
      @keyframes umbrella-pulse {
        0%   { transform: scale(0.55); opacity: 0.75; }
        70%  { transform: scale(1.9);  opacity: 0;    }
        100% { transform: scale(1.9);  opacity: 0;    }
      }
      .umbrella-marker { position: relative; }
      .umbrella-marker-pulse {
        position: absolute; inset: 0; border-radius: 50%;
        animation: umbrella-pulse 2s infinite ease-out;
      }
      .umbrella-marker-badge {
        position: absolute; top: 50%; left: 50%;
        transform: translate(-50%, -50%);
        color: white; border-radius: 50%;
        display: flex; align-items: center; justify-content: center;
        font-family: -apple-system, "Segoe UI", Roboto, sans-serif;
        font-weight: 700; font-size: 13px;
        border: 2.5px solid white;
        box-shadow: 0 3px 8px rgba(0,0,0,0.35);
      }
      .leaflet-tooltip.umbrella-tooltip {
        background: rgba(255,255,255,0.96);
        border: 1px solid #b8c8b8;
        border-radius: 4px;
        padding: 3px 8px;
        font-family: -apple-system, "Segoe UI", Roboto, sans-serif;
        font-size: 11px;
        font-weight: 600;
        color: #1a3a1a;
        box-shadow: 0 2px 4px rgba(0,0,0,0.12);
        white-space: nowrap;
      }
      .leaflet-tooltip.umbrella-tooltip::before { display: none; }
    </style>
    """
    fmap.get_root().header.add_child(folium.Element(css))


def _numbered_pulsing_marker_html(number: int, color: str = "#2C5F2D", size: int = 34) -> str:
    """DivIcon HTML for a numbered marker with a pulsing halo. Uses the
    shared CSS classes injected by ``_inject_shared_css``."""

    inner = size - 12
    return f"""
    <div class="umbrella-marker" style="width:{size}px;height:{size}px;">
      <div class="umbrella-marker-pulse" style="background:{color};"></div>
      <div class="umbrella-marker-badge" style="width:{inner}px;height:{inner}px;background:{color};">
        {number}
      </div>
    </div>
    """


def _add_flyto_animation(fmap, folium, bounds: tuple[list[float], list[float]], duration: float = 1.6) -> None:
    """Inject a JS ``setTimeout`` that calls ``flyToBounds`` on the map
    a fraction of a second after page load. Produces a smooth cinematic
    zoom-in on the observations."""

    sw, ne = bounds
    map_var = fmap.get_name()
    js = f"""
    <script>
      setTimeout(function() {{
        try {{
          {map_var}.flyToBounds(
            [[{sw[0]},{sw[1]}],[{ne[0]},{ne[1]}]],
            {{duration: {duration}, padding: [40, 40]}}
          );
        }} catch (e) {{ /* ignore — map may not be ready yet */ }}
      }}, 350);
    </script>
    """
    fmap.get_root().html.add_child(folium.Element(js))


def _add_legend(fmap, folium, *, title: str, subtitle: str, footer: str = "Sprint 2 — Biodiversity Agent") -> None:
    """Attach a small floating legend card in the bottom-left corner.

    Uses raw HTML injected into the map root — folium's built-in
    ``Marker`` legend is too limited for our per-worker labels.
    """

    html = f"""
    <div style="position: fixed; bottom: 20px; left: 20px; z-index: 9999;
                background: white; padding: 10px 14px; border: 1px solid #d0d0d0;
                border-radius: 8px; font-family: -apple-system, Segoe UI, Roboto, sans-serif;
                font-size: 12px; box-shadow: 0 4px 12px rgba(0,0,0,0.12);
                max-width: 320px;">
      <div style="font-weight: 600; color: #2C5F2D; margin-bottom: 4px;">{title}</div>
      <div style="color: #333; margin-bottom: 6px;">{subtitle}</div>
      <div style="color: #999; font-size: 10px; font-style: italic;">{footer}</div>
    </div>
    """
    fmap.get_root().html.add_child(folium.Element(html))


def _popup_html(rows: list[tuple[str, str]], title: str) -> str:
    """Render a two-column info-card as HTML for a folium popup."""

    body = "".join(
        f'<tr><td style="color:#666;padding:2px 8px 2px 0;font-size:11px;">{k}</td>'
        f'<td style="color:#111;padding:2px 0;font-weight:500;font-size:11px;">{v}</td></tr>'
        for k, v in rows
    )
    return f"""
    <div style="font-family: -apple-system, Segoe UI, Roboto, sans-serif; min-width: 220px;">
      <div style="font-weight:700; color:#2C5F2D; font-size:14px;
                  border-bottom: 1px solid #eee; padding-bottom: 6px; margin-bottom: 6px;">
        {title}
      </div>
      <table style="border-collapse:collapse;">
        {body}
      </table>
    </div>
    """


def _save(fmap, slug: str) -> str:
    target_dir = output_dir()
    out_path = target_dir / f"{slug}.html"
    fmap.save(str(out_path))
    # ``.as_uri()`` is the only cross-platform way to get a valid file URL:
    # Windows produces file:///C:/..., Linux produces file:///path.
    # Using f"file://{path.as_posix()}" produces file://C:/... on Windows,
    # which some URL parsers treat as host="C:" and reject.
    return out_path.resolve().as_uri()


# ============================================================
# 1. Species Distribution — Aziz
# ============================================================


def render_point_map(
    species_name: str,
    coordinates: list[tuple[float, float]],
    metadata: list[dict] | None = None,
    confidence: float | None = None,
    output_dir_override: Path | None = None,
) -> str:
    """Interactive point map for the Species Distribution worker.

    Parameters
    ----------
    species_name
        Scientific name used for the title, legend, and filename.
    coordinates
        Non-empty list of ``(lat, lon)`` observations.
    metadata
        Optional list, one dict per coordinate, containing any of
        ``common_name``, ``country``, ``region``, ``conservation_status``,
        ``year``. Used to populate the rich popup at each marker.
    confidence
        Optional overall confidence 0.0–1.0. Colors the markers.

    Returns
    -------
    ``file://`` URL of the saved HTML.
    """

    if not coordinates:
        raise ValueError("render_point_map requires at least one coordinate.")

    global _OUTPUT_DIR
    if output_dir_override is not None:
        _OUTPUT_DIR = output_dir_override

    slug = _slug(species_name)
    folium = _folium_or_none()
    if folium is None:
        return _placeholder_url(slug)

    from folium.plugins import HeatMap, MarkerCluster

    sw, ne = _bounds(coordinates)
    center = ((sw[0] + ne[0]) / 2, (sw[1] + ne[1]) / 2)

    fmap = _create_base_map(folium, center=center, zoom_start=3)

    color = _confidence_color(confidence)
    metadata = metadata or [{}] * len(coordinates)
    n = len(coordinates)

    # --- Observations layer: styling adapts to dataset density ---
    # ≤ 10   → numbered pulsing badges + permanent tooltip (feels premium
    #          when there are few enough points to read every label)
    # 11-100 → simple CircleMarker + hover tooltip (permanent tooltips
    #          would overlap and the numbering would be meaningless)
    # > 100  → MarkerCluster (points collapse into counted groups when
    #          zoomed out, expand as you zoom in — the standard folium
    #          answer to high-density datasets)
    if n <= 10:
        _add_small_layer(fmap, folium, coordinates, metadata, color, species_name)
    elif n <= 100:
        _add_medium_layer(fmap, folium, coordinates, metadata, color, species_name)
    else:
        _add_large_layer(fmap, folium, coordinates, metadata, color, species_name, MarkerCluster)

    # --- Heatmap toggle (works for any dataset size) ---
    heat_layer = folium.FeatureGroup(name="Heatmap", show=False)
    HeatMap(
        [[lat, lon, 1.0] for lat, lon in coordinates],
        radius=25, blur=18, min_opacity=0.35,
    ).add_to(heat_layer)
    heat_layer.add_to(fmap)

    folium.LayerControl(collapsed=False, position="topright").add_to(fmap)
    fmap.fit_bounds([sw, ne], padding=(30, 30))
    _add_flyto_animation(fmap, folium, ([sw[0], sw[1]], [ne[0], ne[1]]))

    _add_legend(fmap, folium,
                title=species_name,
                subtitle=_build_stats_subtitle(n, confidence, metadata))

    return _save(fmap, slug)


# ---------- adaptive marker layer helpers ----------


def _popup_rows_from_meta(meta: dict, lat: float, lon: float) -> list[tuple[str, str]]:
    """Shared popup row builder used by every marker density branch."""

    rows = [("Location", f"{lat:.3f}°, {lon:.3f}°")]
    if meta.get("common_name"):         rows.append(("Common name",  meta["common_name"]))
    if meta.get("country"):             rows.append(("Country",      meta["country"]))
    if meta.get("region"):              rows.append(("Region",       meta["region"]))
    if meta.get("conservation_status"): rows.append(("IUCN status",  meta["conservation_status"]))
    if meta.get("year"):                rows.append(("Year",         str(meta["year"])))
    return rows


def _add_small_layer(fmap, folium, coordinates, metadata, color, species_name):
    """≤ 10 points — numbered pulsing badges, permanent country tooltips."""

    layer = folium.FeatureGroup(name="Observations", show=True)
    for idx, ((lat, lon), meta) in enumerate(zip(coordinates, metadata), start=1):
        rows = _popup_rows_from_meta(meta, lat, lon)
        country_label = meta.get("country") or meta.get("region") or f"Obs {idx}"
        tooltip = folium.Tooltip(
            f"{idx}. {country_label}",
            permanent=True, direction="right", offset=(14, 0),
            className="umbrella-tooltip",
        )
        folium.Marker(
            location=[lat, lon],
            popup=folium.Popup(_popup_html(rows, species_name), max_width=320),
            tooltip=tooltip,
            icon=folium.DivIcon(
                icon_size=(34, 34), icon_anchor=(17, 17),
                html=_numbered_pulsing_marker_html(idx, color=color),
            ),
        ).add_to(layer)
    layer.add_to(fmap)


def _add_medium_layer(fmap, folium, coordinates, metadata, color, species_name):
    """11-100 points — simple circle markers, tooltip only on hover."""

    layer = folium.FeatureGroup(name="Observations", show=True)
    for (lat, lon), meta in zip(coordinates, metadata):
        rows = _popup_rows_from_meta(meta, lat, lon)
        label = meta.get("country") or meta.get("region") or "Observation"
        folium.CircleMarker(
            location=[lat, lon], radius=6,
            color="#1B3D1C", weight=1, fill=True,
            fill_color=color, fill_opacity=0.85,
            popup=folium.Popup(_popup_html(rows, species_name), max_width=320),
            tooltip=label,
        ).add_to(layer)
    layer.add_to(fmap)


def _add_large_layer(fmap, folium, coordinates, metadata, color, species_name, MarkerCluster):
    """> 100 points — MarkerCluster from folium.plugins. Points collapse
    into counted groups when zoomed out and spiderfy on max zoom."""

    cluster = MarkerCluster(
        name="Observations", show=True,
        options={
            "showCoverageOnHover": False,
            "maxClusterRadius": 55,
            "spiderfyOnMaxZoom": True,
            "disableClusteringAtZoom": 9,
        },
    )
    for (lat, lon), meta in zip(coordinates, metadata):
        rows = _popup_rows_from_meta(meta, lat, lon)
        label = meta.get("country") or meta.get("region") or "Observation"
        folium.CircleMarker(
            location=[lat, lon], radius=5,
            color="#1B3D1C", weight=1, fill=True,
            fill_color=color, fill_opacity=0.75,
            popup=folium.Popup(_popup_html(rows, species_name), max_width=320),
            tooltip=label,
        ).add_to(cluster)
    cluster.add_to(fmap)


def _build_stats_subtitle(n: int, confidence: float | None, metadata: list[dict]) -> str:
    """Build the HTML fragment shown as the legend subtitle. Includes the
    top 3 countries when the metadata carries country info — the country
    breakdown is what turns a wall of 300 dots into an actual insight."""

    from collections import Counter

    parts = [f"{n} observation{'s' if n != 1 else ''}"]
    if confidence is not None:
        parts.append(f"confidence {confidence:.2f}")
    subtitle = " · ".join(parts)

    country_counts = Counter(m.get("country") for m in metadata if m.get("country"))
    if country_counts:
        top = country_counts.most_common(3)
        rows_html = "".join(
            f"<div style='display:flex;justify-content:space-between;"
            f"padding:1px 0;font-size:11px;'>"
            f"<span style='color:#555;'>{c}</span>"
            f"<span style='color:#2C5F2D;font-weight:700;'>{cnt}</span>"
            f"</div>"
            for c, cnt in top
        )
        subtitle += (
            "<div style='margin-top:6px;padding-top:6px;"
            "border-top:1px solid #eee;'>"
            "<div style='font-size:10px;color:#999;letter-spacing:1.5px;"
            "text-transform:uppercase;margin-bottom:3px;'>Top countries</div>"
            f"{rows_html}"
            "</div>"
        )
    return subtitle


# ============================================================
# 2. Habitat Visualization — Ouissale
# ============================================================


def render_habitat_map(
    species_name: str,
    regions: list[dict],
    conservation_status: str | None = None,
    output_dir_override: Path | None = None,
) -> str:
    """Polygon overlay map of a species' habitat regions.

    Parameters
    ----------
    species_name
        For title and filename.
    regions
        List of dicts, each with:
        ``{"name": str, "bounds": [[sw_lat, sw_lon], [ne_lat, ne_lon]],
           "description": str (optional)}``
        A rectangle overlay is drawn per region. Real Sprint 3 workers
        should pass actual GeoJSON polygons instead — the same function
        will handle those via ``folium.GeoJson``.
    conservation_status
        Optional IUCN status ("Endangered", "Vulnerable", ...). Colors
        every region accordingly.
    """

    if not regions:
        raise ValueError("render_habitat_map requires at least one region.")

    global _OUTPUT_DIR
    if output_dir_override is not None:
        _OUTPUT_DIR = output_dir_override

    slug = f"habitat_{_slug(species_name)}"
    folium = _folium_or_none()
    if folium is None:
        return _placeholder_url(slug)

    status_colors = {
        "Extinct":            "#4A148C",
        "Critically Endangered": "#B71C1C",
        "Endangered":         "#D32F2F",
        "Vulnerable":         "#EF6C00",
        "Near Threatened":    "#F9A825",
        "Least Concern":      "#2E7D32",
    }
    color = status_colors.get(conservation_status or "", "#2C5F2D")

    # Aggregate all rectangle bounds to compute the overall map extent.
    all_bounds = []
    for r in regions:
        b = r.get("bounds")
        if b:
            all_bounds.extend(b)
    sw, ne = _bounds(all_bounds) if all_bounds else ([0, 0], [0, 0])
    center = ((sw[0] + ne[0]) / 2, (sw[1] + ne[1]) / 2)

    fmap = _create_base_map(folium, center=center, zoom_start=2)

    habitat_layer = folium.FeatureGroup(name="Habitat regions", show=True)
    for r in regions:
        if not r.get("bounds"):
            continue
        rows = [("Habitat region", r["name"])]
        if conservation_status:
            rows.append(("IUCN status", conservation_status))
        if r.get("description"):
            rows.append(("Notes", r["description"]))

        folium.Rectangle(
            bounds=r["bounds"],
            color=color,
            weight=2,
            fill=True,
            fill_color=color,
            fill_opacity=0.28,
            popup=folium.Popup(_popup_html(rows, species_name), max_width=320),
            tooltip=r["name"],
        ).add_to(habitat_layer)
    habitat_layer.add_to(fmap)

    folium.LayerControl(collapsed=False, position="topright").add_to(fmap)
    if all_bounds:
        fmap.fit_bounds([sw, ne], padding=(30, 30))
        _add_flyto_animation(fmap, folium, ([sw[0], sw[1]], [ne[0], ne[1]]))

    _add_legend(fmap, folium,
                title=species_name,
                subtitle=f"{len(regions)} habitat region{'s' if len(regions) != 1 else ''}"
                         + (f" · {conservation_status}" if conservation_status else ""))
    return _save(fmap, slug)


# ============================================================
# 3. Biodiversity Hotspots — Ouissale
# ============================================================


def render_heatmap(
    region: str,
    hotspots: list[dict],
    output_dir_override: Path | None = None,
) -> str:
    """Density heatmap + hotspot markers for the Biodiversity Hotspots worker.

    Parameters
    ----------
    region
        Region label for title and filename ("global", "africa", ...).
    hotspots
        List of dicts: ``{"name": str, "center": (lat, lon),
        "species_richness": int}``.
    """

    if not hotspots:
        raise ValueError("render_heatmap requires at least one hotspot.")

    global _OUTPUT_DIR
    if output_dir_override is not None:
        _OUTPUT_DIR = output_dir_override

    slug = f"hotspots_{_slug(region)}"
    folium = _folium_or_none()
    if folium is None:
        return _placeholder_url(slug)

    from folium.plugins import HeatMap

    centers = [h["center"] for h in hotspots if h.get("center")]
    sw, ne = _bounds(centers)
    center = ((sw[0] + ne[0]) / 2, (sw[1] + ne[1]) / 2)

    fmap = _create_base_map(folium, center=center, zoom_start=2)

    # --- Heatmap layer weighted by species_richness ---
    max_richness = max((h.get("species_richness", 1) for h in hotspots), default=1)
    heat_data = [
        [h["center"][0], h["center"][1], h.get("species_richness", 1) / max_richness]
        for h in hotspots
        if h.get("center")
    ]
    heat_layer = folium.FeatureGroup(name="Density heatmap", show=True)
    HeatMap(heat_data, radius=45, blur=35, min_opacity=0.4).add_to(heat_layer)
    heat_layer.add_to(fmap)

    # --- Named hotspot markers overlay (numbered + pulsing) ---
    hs_layer = folium.FeatureGroup(name="Hotspot centers", show=True)
    for idx, h in enumerate(hotspots, start=1):
        if not h.get("center"):
            continue
        lat, lon = h["center"]
        rows = [
            ("Hotspot",          h["name"]),
            ("Species richness", str(h.get("species_richness", "n/a"))),
            ("Center",           f"{lat:.2f}°, {lon:.2f}°"),
        ]
        tooltip = folium.Tooltip(
            f"{idx}. {h['name']}",
            permanent=True, direction="right", offset=(14, 0),
            className="umbrella-tooltip",
        )
        folium.Marker(
            location=[lat, lon],
            popup=folium.Popup(_popup_html(rows, "Biodiversity hotspot"), max_width=320),
            tooltip=tooltip,
            icon=folium.DivIcon(
                icon_size=(34, 34), icon_anchor=(17, 17),
                html=_numbered_pulsing_marker_html(idx, color="#2E7D32"),
            ),
        ).add_to(hs_layer)
    hs_layer.add_to(fmap)

    folium.LayerControl(collapsed=False, position="topright").add_to(fmap)
    fmap.fit_bounds([sw, ne], padding=(40, 40))
    _add_flyto_animation(fmap, folium, ([sw[0], sw[1]], [ne[0], ne[1]]))

    _add_legend(fmap, folium,
                title=f"Biodiversity hotspots — {region}",
                subtitle=f"{len(hotspots)} hotspots · richness weighted heatmap")
    return _save(fmap, slug)


# ============================================================
# 4. Migration Analysis — Mariem
# ============================================================


def render_route_map(
    species_name: str,
    route: list[tuple[float, float]],
    seasonal_pattern: str | None = None,
    output_dir_override: Path | None = None,
) -> str:
    """Migration route map with numbered waypoints and directional flow.

    Parameters
    ----------
    species_name
        Scientific name for title and filename.
    route
        Ordered list of ``(lat, lon)`` waypoints — the polyline is drawn
        in order and each waypoint gets a numbered popup.
    seasonal_pattern
        Optional short label displayed in the legend ("Pole-to-pole",
        "Wet/dry seasonal").
    """

    if len(route) < 2:
        raise ValueError("render_route_map requires at least two waypoints.")

    global _OUTPUT_DIR
    if output_dir_override is not None:
        _OUTPUT_DIR = output_dir_override

    slug = f"migration_{_slug(species_name)}"
    folium = _folium_or_none()
    if folium is None:
        return _placeholder_url(slug)

    from folium.plugins import PolyLineTextPath

    sw, ne = _bounds(route)
    center = ((sw[0] + ne[0]) / 2, (sw[1] + ne[1]) / 2)

    fmap = _create_base_map(folium, center=center, zoom_start=2)

    # --- Route polyline with directional arrows ---
    route_layer = folium.FeatureGroup(name="Migration route", show=True)
    line = folium.PolyLine(
        locations=[list(pt) for pt in route],
        color="#2C5F2D",
        weight=4,
        opacity=0.85,
    )
    line.add_to(route_layer)
    # Add arrow glyphs along the polyline (folium plugin) to show direction.
    PolyLineTextPath(
        line, "  ▶  ", repeat=True,
        offset=6, attributes={"fill": "#2C5F2D", "font-weight": "700", "font-size": "16"},
    ).add_to(route_layer)
    route_layer.add_to(fmap)

    # --- Numbered pulsing waypoints with popups + permanent tooltips ---
    waypoint_layer = folium.FeatureGroup(name="Waypoints", show=True)
    for i, (lat, lon) in enumerate(route, start=1):
        role = ("Start" if i == 1 else "End" if i == len(route) else f"Stop {i - 1}")
        rows = [
            ("Waypoint", f"{i} of {len(route)}"),
            ("Role",     role),
            ("Location", f"{lat:.3f}°, {lon:.3f}°"),
        ]
        tooltip = folium.Tooltip(
            f"{i}. {role}",
            permanent=True, direction="right", offset=(14, 0),
            className="umbrella-tooltip",
        )
        folium.Marker(
            location=[lat, lon],
            popup=folium.Popup(_popup_html(rows, species_name), max_width=320),
            tooltip=tooltip,
            icon=folium.DivIcon(
                icon_size=(34, 34), icon_anchor=(17, 17),
                html=_numbered_pulsing_marker_html(i, color="#2C5F2D"),
            ),
        ).add_to(waypoint_layer)
    waypoint_layer.add_to(fmap)

    folium.LayerControl(collapsed=False, position="topright").add_to(fmap)
    fmap.fit_bounds([sw, ne], padding=(30, 30))
    _add_flyto_animation(fmap, folium, ([sw[0], sw[1]], [ne[0], ne[1]]))

    _add_legend(fmap, folium,
                title=species_name,
                subtitle=f"{len(route)} waypoints"
                         + (f" · {seasonal_pattern}" if seasonal_pattern else ""))
    return _save(fmap, slug)
