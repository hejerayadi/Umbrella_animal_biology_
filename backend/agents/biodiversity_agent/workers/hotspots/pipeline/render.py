"""M3 step 8 - the render specification, plus an optional folium rendering.

Doc §2.6 is explicit that M3 emits a *specification* and the frontend draws it,
so ``build_render_spec`` is the contract-level output. ``render_folium`` is an
extra convenience for the dashboard and the defence: it writes a real
interactive map to disk and returns its path, which is also what makes the
result presentable without the React frontend running.
"""

from __future__ import annotations

import math

import pandas as pd

from ...common.config import OUTPUT_DIR
from ..schema import LegendEntry, MapRenderSpec
from .ranking import quantile_breaks

# Yellow (few species) to dark red (many) - the YlOrRd ramp of the notebook.
PALETTE = ["#ffffcc", "#ffeda0", "#fed976", "#feb24c", "#fd8d3c", "#f03b20", "#bd0026"]

# One degree of latitude, and of longitude at the equator - for turning a cell
# size in kilometres back into the rectangle a map can draw.
KM_PER_DEGREE_LAT = 110.57
KM_PER_DEGREE_LON = 111.32
EARTH_RADIUS_KM = 6371.0        # the same radius grid.assign_cells projects with

# Above this, only the richest cells get their own rectangle - see render_folium.
MAX_DRAWN_CELLS = 1_500


def build_render_spec(cells: pd.DataFrame,
                      bbox: tuple[float, float, float, float]) -> MapRenderSpec:
    """The heatmap specification: quantile scale, seven classes, legend."""

    breaks = quantile_breaks(cells["corrected_species"].to_numpy(), classes=7) \
        if not cells.empty else []

    legend = [
        LegendEntry(label=f"{breaks[i]:g} - {breaks[i + 1]:g}",
                    colour_hex=PALETTE[i], value=breaks[i + 1])
        for i in range(len(breaks) - 1)
    ] if breaks else []

    center = ((round(float(cells["lat"].mean()), 3), round(float(cells["lon"].mean()), 3))
              if not cells.empty else None)

    return MapRenderSpec(
        type="heatmap",
        scale="quantile",
        classes=7,
        bbox=bbox,
        center=center,
        zoom=5,
        legend=legend,
    )


def _colour_for(value: float, breaks: list[float]) -> str:
    """The palette colour for one cell, by the same quantile breaks as the legend."""

    if not breaks:
        return PALETTE[0]
    for index in range(len(breaks) - 1):
        if value <= breaks[index + 1]:
            return PALETTE[min(index, len(PALETTE) - 1)]
    return PALETTE[-1]


def _cell_bounds(cell_id: str, cell_size_km: float, lat: float, lon: float):
    """The true corners of one grid cell, recovered from its id.

    ``grid.assign_cells`` builds the id from the equal-area sinusoidal
    projection - ``x = R * lon * cos(lat)``, ``y = R * lat``, floored by the cell
    size - so the id *is* the geometry and can be inverted exactly.

    Drawing a box around the mean of a cell's records instead, which is what this
    did first, puts each rectangle wherever its records happen to sit: the boxes
    overlap, sit off the cell they describe, and stop tiling. Malta showed it
    plainly - a dozen translucent squares layered over each other.

    ``lat``/``lon`` are the fallback for an id this cannot parse.
    """

    try:
        column, row = (int(part) for part in str(cell_id).split("_"))
    except (TypeError, ValueError):
        half_lat = (cell_size_km / 2) / KM_PER_DEGREE_LAT
        cosine = max(math.cos(math.radians(lat)), 0.1)
        half_lon = (cell_size_km / 2) / (KM_PER_DEGREE_LON * cosine)
        return [[lat - half_lat, lon - half_lon], [lat + half_lat, lon + half_lon]]

    south = math.degrees(row * cell_size_km / EARTH_RADIUS_KM)
    north = math.degrees((row + 1) * cell_size_km / EARTH_RADIUS_KM)
    # The projection uses the cosine of the record's own latitude; the cell's
    # mid-latitude is the right inverse for drawing its width.
    cosine = max(math.cos(math.radians((south + north) / 2)), 0.05)
    west = math.degrees(column * cell_size_km / (EARTH_RADIUS_KM * cosine))
    east = math.degrees((column + 1) * cell_size_km / (EARTH_RADIUS_KM * cosine))
    return [[south, west], [north, east]]


def _legend_html(breaks: list[float], cell_size_km: float, effort: bool,
                 *, shown: int = 0, total: int = 0) -> str:
    """The colour key, drawn on the map itself.

    A heat layer without a key is decoration: the reader can see that one place is
    redder than another but not what red means, or in what units.
    """

    if not breaks:
        return ""
    rows = []
    for index in range(len(breaks) - 1):
        rows.append(
            f'<div style="display:flex;align-items:center;gap:6px;">'
            f'<span style="width:14px;height:10px;background:{PALETTE[index]};'
            f'border:1px solid #999;"></span>'
            f'<span>{breaks[index]:g} &ndash; {breaks[index + 1]:g}</span></div>')
    measure = ("species per cell, corrected for recording effort" if effort
               else "species per cell (uncorrected)")
    return (
        '<div style="position:fixed;bottom:18px;left:12px;z-index:9999;'
        'background:rgba(255,255,255,.94);padding:8px 10px;border-radius:6px;'
        'border:1px solid #bbb;font:11px/1.45 system-ui,sans-serif;color:#222;'
        'box-shadow:0 1px 4px rgba(0,0,0,.25);">'
        f'<div style="font-weight:600;margin-bottom:4px;">{measure}</div>'
        + "".join(rows)
        + f'<div style="margin-top:5px;color:#555;">one dot per '
          f'{cell_size_km:g} km cell &middot; bigger and darker = more species '
          f'&middot; hover for its figures'
        + ('' if not total or shown >= total
           else f' &middot; richest {shown:,} of {total:,} drawn')
        + '</div></div>')


def render_folium(cells: pd.DataFrame, clusters, region_slug: str, *,
                  cell_size_km: float = 42.0,
                  effort_applied: bool = True) -> str | None:
    """Write an interactive map and return its path, or None if folium is absent.

    Four layers, and the order matters for what a reader can do with it:

    * **the cells themselves**, one rectangle each, coloured by the same quantile
      breaks as the legend and carrying their own figures on hover. This is what
      makes the map answer "why is this square dark?" - the heat layer alone
      cannot, because it interpolates between points and belongs to no cell;
    * **the heat layer**, kept for the overall shape, off by default now that the
      cells carry the detail;
    * **a circle per hotspot**, showing its extent;
    * **a marker per hotspot**, with its ranked figures.

    The colour key is drawn onto the map, since a gradient without units is
    decoration.
    """

    if cells.empty:
        return None

    try:
        import folium
        from folium.plugins import Fullscreen, HeatMap
    except ImportError:
        return None

    fmap = folium.Map(
        location=[float(cells["lat"].mean()), float(cells["lon"].mean())],
        zoom_start=5,
        tiles="OpenStreetMap",
    )

    breaks = quantile_breaks(cells["corrected_species"].to_numpy(), classes=7)

    # --- one mark per cell, hoverable and clickable
    #
    # Two ways of drawing the same cells. Points are the default because a grid of
    # tiles is hard to read - it hides the basemap and every cell shouts equally
    # loudly. A point carries the value twice, in colour and in size, so the eye
    # finds the rich cells without a legend. The tiles stay one click away for
    # anyone who wants to see the actual footprint that was measured.
    point_layer = folium.FeatureGroup(name="Cells as points", show=True)
    tile_layer = folium.FeatureGroup(name="Cells as tiles (true footprint)",
                                    show=False)

    values = cells["corrected_species"].astype(float)
    lowest, highest = float(values.min()), float(values.max())
    spread = (highest - lowest) or 1.0

    drawn = cells
    if len(cells) > MAX_DRAWN_CELLS:
        drawn = cells.nlargest(MAX_DRAWN_CELLS, "corrected_species")

    for _, cell in drawn.iterrows():
        corrected = float(cell["corrected_species"])
        cluster = cell.get("cluster_id")
        belongs = ("unassigned - too sparse to join a hotspot"
                   if cluster is None or cluster != cluster else
                   f"Hotspot {int(cluster)}")
        factor = float(cell.get("effort_factor", 1.0))
        effort_line = (f"{factor:.2f}x effort correction"
                       if abs(factor - 1.0) > 0.005 else "no effort adjustment")
        colour = _colour_for(corrected, breaks)
        bounds = _cell_bounds(cell["cell_id"], cell_size_km,
                              float(cell["lat"]), float(cell["lon"]))
        centre = [(bounds[0][0] + bounds[1][0]) / 2,
                  (bounds[0][1] + bounds[1][1]) / 2]

        tooltip = (f"{corrected:.0f} species (corrected) &middot; "
                   f"{int(cell['n_species'])} recorded &middot; {belongs}")
        popup = folium.Popup(
            f"<b>Cell {cell['cell_id']}</b><br>"
            f"species recorded: {int(cell['n_species'])}<br>"
            f"after effort correction: {corrected:.1f}<br>"
            f"records: {int(cell['n_records'])}<br>"
            f"Shannon: {float(cell['shannon']):.2f} &middot; "
            f"Simpson: {float(cell['simpson']):.2f}<br>"
            f"{effort_line}<br>"
            f"country: {cell.get('country') or 'n/a'}<br>"
            f"{belongs}",
            max_width=300)

        folium.CircleMarker(
            location=centre,
            radius=5 + 9 * ((corrected - lowest) / spread),
            color="#33333366", weight=1,
            fill=True, fill_color=colour, fill_opacity=0.85,
            tooltip=tooltip, popup=popup,
        ).add_to(point_layer)

        folium.Rectangle(
            bounds=bounds,
            color=None, weight=0, opacity=0,
            fill=True, fill_color=colour, fill_opacity=0.45,
            tooltip=tooltip,
        ).add_to(tile_layer)

    point_layer.add_to(fmap)
    tile_layer.add_to(fmap)

    # --- the heat layer keeps the overall shape, but the cells carry the detail,
    #     so it starts hidden. `show=False` on the plugin itself is not honoured -
    #     it arrives checked - so the switch lives on a FeatureGroup around it.
    heat_layer = folium.FeatureGroup(name="Heat blur (overall shape)", show=False)
    HeatMap(
        cells[["lat", "lon", "corrected_species"]].values.tolist(),
        radius=20, blur=25, min_opacity=0.3,
    ).add_to(heat_layer)
    heat_layer.add_to(fmap)

    hotspot_layer = folium.FeatureGroup(name="Ranked hotspots", show=True)
    for cluster in clusters:
        lat, lon = cluster.centroid
        # A circle whose radius reflects the cluster's area, so extent is visible.
        folium.Circle(
            location=[lat, lon],
            radius=(cluster.area_km2 ** 0.5) * 500,
            color="darkred", weight=2, fill=True, fill_opacity=0.08,
        ).add_to(hotspot_layer)

        example = cluster.top_species[0].scientific_name if cluster.top_species else "?"
        folium.Marker(
            location=[lat, lon],
            popup=folium.Popup(
                f"<b>#{cluster.rank} {cluster.label}</b><br>"
                f"species: {cluster.species_count:,}<br>"
                f"records: {cluster.record_count:,}<br>"
                f"area: {cluster.area_km2:,.0f} km2<br>"
                f"cells: {cluster.cell_count}<br>"
                f"top species: {example}",
                max_width=320),
            tooltip=f"#{cluster.rank} {cluster.label} - {cluster.species_count:,} species",
            icon=folium.Icon(color="red", icon="leaf"),
        ).add_to(hotspot_layer)
    hotspot_layer.add_to(fmap)

    folium.LayerControl(collapsed=True).add_to(fmap)
    Fullscreen(position="topright", title="Enlarge",
               title_cancel="Back", force_separate_button=True).add_to(fmap)
    legend = _legend_html(breaks, cell_size_km, effort_applied,
                          shown=len(drawn), total=len(cells))
    if legend:
        fmap.get_root().html.add_child(folium.Element(legend))

    out_path = OUTPUT_DIR / f"hotspots_{region_slug}.html"
    fmap.save(str(out_path))
    return str(out_path)

def render_species_folium(points: pd.DataFrame, clusters, slug: str) -> str | None:
    """A map for the species path: the records themselves, plus their clusters.

    The heat layer here is observation density, not richness, so it carries no
    richness legend - it would be a legend for a quantity this path never
    computes.
    """

    if points.empty:
        return None

    try:
        import folium
        from folium.plugins import Fullscreen, HeatMap
    except ImportError:
        return None

    lat = points["decimalLatitude"].astype(float)
    lon = points["decimalLongitude"].astype(float)

    fmap = folium.Map(location=[float(lat.mean()), float(lon.mean())],
                      zoom_start=4, tiles="OpenStreetMap")

    HeatMap([[a, b, 1.0] for a, b in zip(lat, lon)],
            radius=14, blur=20, min_opacity=0.25,
            name="Occurrence density").add_to(fmap)

    for cluster in clusters:
        centre_lat, centre_lon = cluster.centroid
        folium.Circle(
            location=[centre_lat, centre_lon],
            radius=max((cluster.area_km2 ** 0.5) * 500, 20_000),
            color="darkblue", weight=2, fill=True, fill_opacity=0.06,
        ).add_to(fmap)
        folium.Marker(
            location=[centre_lat, centre_lon],
            popup=folium.Popup(
                f"<b>#{cluster.rank} {cluster.label}</b><br>"
                f"records: {cluster.record_count:,}<br>"
                f"extent: {cluster.area_km2:,.0f} km2",
                max_width=320),
            tooltip=f"#{cluster.rank} - {cluster.record_count:,} records",
            icon=folium.Icon(color="blue", icon="paw-print", prefix="fa"),
        ).add_to(fmap)

    folium.LayerControl(collapsed=True).add_to(fmap)
    Fullscreen(position="topright", title="Enlarge",
               title_cancel="Back", force_separate_button=True).add_to(fmap)
    # The same reasoning as the hotspot map: a gradient with no units is
    # decoration. Here the quantity is records, not species.
    fmap.get_root().html.add_child(folium.Element(
        '<div style="position:fixed;bottom:18px;left:12px;z-index:9999;'
        'background:rgba(255,255,255,.94);padding:8px 10px;border-radius:6px;'
        'border:1px solid #bbb;font:11px/1.45 system-ui,sans-serif;color:#222;'
        'box-shadow:0 1px 4px rgba(0,0,0,.25);">'
        '<div style="font-weight:600;margin-bottom:3px;">density of records</div>'
        '<div>warmer = more records of this species in that spot</div>'
        '<div style="margin-top:4px;color:#555;">this is observation density, '
        'not abundance &middot; markers are ranked clusters, hover for their '
        'figures</div></div>'))
    out_path = OUTPUT_DIR / f"occurrences_{slug}.html"
    fmap.save(str(out_path))
    return str(out_path)
