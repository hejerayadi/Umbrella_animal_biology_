"""M3 stage T2 - bin. Fold coordinates onto an equal-area grid.

Doc §5.2 stage T2, and the direct answer to problem P3.

The document specifies H3 hexagons. The ``h3`` library is not installed here, so
equal-area sinusoidal squares are used instead and the substitution is reported.
What must not change is the *equal-area* property: a one-degree cell covers about
12 300 km2 at the equator and about 3 100 km2 at 75 deg N, so a degree grid would
report an artefact of latitude alongside the biology.

    x = R * lon_radians * cos(lat)      y = R * lat_radians

The cosine is the whole point: towards the pole it shrinks the cell exactly as
the real Earth does, so every cell ends up covering the same ground.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ...common.config import EARTH_RADIUS_KM


def assign_cells(table: pd.DataFrame, cell_size_km: float) -> pd.DataFrame:
    """Add a ``cell_id`` column naming the equal-area cell of each record."""

    if table.empty:
        out = table.copy()
        out["cell_id"] = []
        return out

    lat_rad = np.radians(table["decimalLatitude"].to_numpy())
    lon_rad = np.radians(table["decimalLongitude"].to_numpy())

    x_km = EARTH_RADIUS_KM * lon_rad * np.cos(lat_rad)
    y_km = EARTH_RADIUS_KM * lat_rad

    # floor() rather than round(): a cell is a half-open interval, so a point on
    # a boundary belongs to exactly one cell.
    column = np.floor(x_km / cell_size_km).astype(int)
    row = np.floor(y_km / cell_size_km).astype(int)

    out = table.copy()
    out["cell_id"] = [f"{c}_{r}" for c, r in zip(column, row)]
    return out


def study_area_km2(bbox: tuple[float, float, float, float]) -> float:
    """Area of the study box on an equal-area footing, in km2.

    Integrating cos(lat) over the latitude span is what keeps this honest; the
    naive width x height in degrees would overstate a tropical box and
    understate a polar one.
    """

    lon_min, lat_min, lon_max, lat_max = bbox
    lon_span_rad = np.radians(abs(lon_max - lon_min))
    lat_min_rad, lat_max_rad = np.radians(lat_min), np.radians(lat_max)
    return float(
        EARTH_RADIUS_KM ** 2 * lon_span_rad * abs(np.sin(lat_max_rad) - np.sin(lat_min_rad))
    )
