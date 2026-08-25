"""Species-scoped clustering: where one species' records concentrate.

The region-scoped pipeline bins records onto a grid, measures richness per cell,
corrects for effort and clusters the cells. None of that applies to a single
species: richness per cell is 1 everywhere, so the grid carries no information
and the effort correction has nothing to normalise.

So this path clusters the occurrence points themselves and ranks the clusters by
how many records they hold. That is **observation density**, not biodiversity -
the worker labels it as such, and says so in the summary, because a cluster here
can be one well-photographed reserve rather than a stronghold of the species.

The three DBSCAN correctness requirements are the same as in ``clustering.py``:
haversine, eps in radians, ball_tree. What differs is the absence of
``sample_weight`` - every record is one observation, and there is no richness to
weight by.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN

from ...common.config import M3_EPS_GRID, M3_MIN_SAMPLES_GRID, M3_NOISE_BAND
from ..schema import HotspotCluster, SpeciesCount
from .clustering import measure_quality

EARTH_RADIUS_KM = 6371.0

# One degree of latitude, and of longitude at the equator. Longitude shrinks by
# cos(latitude), which is why the extent below multiplies by it.
KM_PER_DEGREE_LAT = 110.57
KM_PER_DEGREE_LON = 111.32

# The silhouette is quadratic in pairwise distances, so scoring 28 candidate
# configurations against every occurrence point costs more than the fits do.
# The scan scores a fixed sample; the reported silhouette, computed once, uses
# every point.
SCAN_SAMPLE = 1_000


def as_latlon(points: pd.DataFrame) -> pd.DataFrame:
    """The lat/lon column names ``measure_quality`` and the renderers expect."""

    return pd.DataFrame({
        "lat": points["decimalLatitude"].to_numpy(dtype=float),
        "lon": points["decimalLongitude"].to_numpy(dtype=float),
    })


def cluster_occurrences(points: pd.DataFrame, eps_km: float, min_samples: int):
    """Fit DBSCAN on the occurrence coordinates themselves."""

    positions = np.radians(as_latlon(points)[["lat", "lon"]].to_numpy())
    model = DBSCAN(
        eps=eps_km / EARTH_RADIUS_KM,
        min_samples=min_samples,
        metric="haversine",
        algorithm="ball_tree",
    ).fit(positions)
    return model, model.labels_


def tune_occurrences(points: pd.DataFrame, fallback_eps: float,
                     fallback_min_samples: int):
    """The same explicit scan as the region path, over occurrence points.

    Still no GridSearchCV: there are no labels to score against. Silhouette
    decides, with the noise share held inside the documented band so a
    configuration that rejects everything - or nothing - cannot win.
    """

    low, high = M3_NOISE_BAND
    frame = as_latlon(points)
    rows: list[dict] = []
    best: tuple[float, float, int] | None = None

    for eps_km in M3_EPS_GRID:
        for min_samples in M3_MIN_SAMPLES_GRID:
            if len(points) <= min_samples:
                continue
            _, labels = cluster_occurrences(points, eps_km, min_samples)
            quality = measure_quality(frame, labels, eps_km=eps_km,
                                      min_samples=min_samples,
                                      sample_size=SCAN_SAMPLE)
            rows.append({
                "eps_km": eps_km,
                "min_samples": min_samples,
                "n_clusters": quality.n_clusters,
                "noise_share": quality.noise_share,
                "silhouette": quality.silhouette,
            })
            admissible = (quality.n_clusters >= 2
                          and quality.silhouette is not None
                          and low <= quality.noise_share <= high)
            if admissible and (best is None or quality.silhouette > best[0]):
                best = (quality.silhouette, eps_km, min_samples)

    if best is None:
        return fallback_eps, fallback_min_samples, rows, False
    return best[1], best[2], rows, True


def extent_km2(subset: pd.DataFrame) -> float:
    """The area of a cluster's bounding box, in square kilometres.

    A bounding box, not a convex hull: the hull would need scipy for a value
    nobody reads to two significant figures, and the box is the honest
    over-estimate. A single-point cluster has no extent, hence the 0.0.
    """

    if len(subset) < 2:
        return 0.0
    lat = subset["decimalLatitude"].to_numpy(dtype=float)
    lon = subset["decimalLongitude"].to_numpy(dtype=float)
    mean_lat = float(np.mean(lat))
    height = (lat.max() - lat.min()) * KM_PER_DEGREE_LAT
    width = (lon.max() - lon.min()) * KM_PER_DEGREE_LON * np.cos(np.radians(mean_lat))
    return round(float(abs(height * width)), 1)


def build_occurrence_clusters(points: pd.DataFrame, labels: np.ndarray,
                              *, top_n: int, species_name: str
                              ) -> tuple[list[HotspotCluster], list[HotspotCluster]]:
    """Describe and rank the clusters by record count.

    Record count is the only ranking that means anything for one species. It is
    reported as an occurrence count, never as richness.
    """

    table = points.copy()
    table["cluster_id"] = [None if label == -1 else int(label) for label in labels]
    assigned = table[table["cluster_id"].notna()]
    if assigned.empty:
        return [], []

    clusters: list[HotspotCluster] = []
    busiest = int(assigned["cluster_id"].value_counts().max())

    for cluster_id, subset in assigned.groupby("cluster_id"):
        countries = subset["countryCode"].dropna()
        dominant = str(countries.mode().iloc[0]) if not countries.empty else None
        years = subset["year"].dropna()
        clusters.append(HotspotCluster(
            cluster_id=int(cluster_id),
            # Named for what it is. Calling this a biodiversity hotspot would be
            # false: it is where this one species has been recorded.
            label=f"Occurrence cluster {int(cluster_id)}"
                  + (f" ({dominant})" if dominant else ""),
            centroid=(round(float(subset["decimalLatitude"].mean()), 3),
                      round(float(subset["decimalLongitude"].mean()), 3)),
            area_km2=extent_km2(subset),
            cell_count=0,                       # no grid on this path
            species_count=int(subset["speciesKey"].nunique()),
            record_count=int(len(subset)),
            score=round(len(subset) / busiest, 3) if busiest else 0.0,
            rank=0,
            top_species=[SpeciesCount(
                species_key=int(subset["speciesKey"].iloc[0]),
                scientific_name=species_name,
                record_count=int(len(subset)))],
            dominant_country=dominant,
        ))

    ranked = sorted(clusters, key=lambda c: c.record_count, reverse=True)
    for position, cluster in enumerate(ranked, start=1):
        cluster.rank = position
    return clusters, ranked[:top_n]
