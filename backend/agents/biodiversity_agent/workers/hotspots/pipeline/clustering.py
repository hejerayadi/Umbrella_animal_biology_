"""M3 step 6 - cluster. DBSCAN over cell centroids.

Doc §4.4. DBSCAN is the algorithm of choice for three reasons that matter more
than raw accuracy: it does not need the number of clusters in advance (nobody
knows how many hotspots a region has), it finds arbitrarily shaped clusters (a
hotspot following a river is not a circle, which rules out k-means), and it
labels sparse cells as noise instead of forcing every cell into a cluster - so
the Sahara does not become a hotspot.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors

from ...common.config import (
    EARTH_RADIUS_KM,
    M3_EPS_GRID,
    M3_MIN_SAMPLES_GRID,
    M3_NOISE_BAND,
)
from ..schema import ClusteringQuality


def _radians(cells: pd.DataFrame) -> np.ndarray:
    """Cell centroids as radians, in (lat, lon) order.

    Deliberately NOT standardised. TP3 scales before KNN and TP5 before k-means
    because those measure distances in feature space; latitude and longitude are
    already a physical position, and standardising them destroys the meaning of
    the distance. The correct treatment is the haversine metric on raw radians.
    """

    return np.radians(cells[["lat", "lon"]].to_numpy())


def k_distance_curve(cells: pd.DataFrame, k: int) -> np.ndarray:
    """Sorted distance to the k-th nearest neighbour, in km.

    DBSCAN's counterpart to the elbow method of TP5: the knee of this curve
    separates cells in dense zones from isolated ones, which is what eps has to
    sit at.
    """

    if len(cells) <= k:
        return np.array([])

    positions = _radians(cells)
    finder = NearestNeighbors(n_neighbors=k, metric="haversine", algorithm="ball_tree")
    finder.fit(positions)
    distances, _ = finder.kneighbors(positions)
    return np.sort(distances[:, -1]) * EARTH_RADIUS_KM


def run_dbscan(cells: pd.DataFrame, eps_km: float, min_samples: int):
    """Fit DBSCAN, weighting each cell by its corrected richness.

    Three lines are correctness requirements, not tuning options:

    * ``metric="haversine"`` - degrees are not distances. Euclidean distance over
      lat/lon is wrong everywhere and badly wrong far from the equator.
    * ``eps / EARTH_RADIUS_KM`` - scikit-learn works in radians for haversine, so
      kilometres must be divided by the Earth radius.
    * ``algorithm="ball_tree"`` - the only scikit-learn algorithm supporting
      haversine.

    And the weights are normalised by their median. scikit-learn treats a sample
    whose weight is at least ``min_samples`` as a core sample on its own;
    corrected richness runs from roughly 12 to 81, so passing it raw - as the
    design document's own snippet in §4.4 does - makes EVERY cell a core sample.
    The noise label becomes unreachable, DBSCAN silently degrades into connected
    components at radius eps, and the effort correction stops affecting the
    labels at all. Dividing by the median makes a median-richness cell count as
    exactly one sample, which restores ``min_samples`` to its plain meaning.
    """

    positions = _radians(cells)

    weights = cells["corrected_species"].to_numpy(dtype=float)
    median = float(np.median(weights))
    if median > 0:
        weights = weights / median

    model = DBSCAN(
        eps=eps_km / EARTH_RADIUS_KM,
        min_samples=min_samples,
        metric="haversine",
        algorithm="ball_tree",
    ).fit(positions, sample_weight=weights)

    return model, model.labels_


def measure_quality(cells: pd.DataFrame, labels: np.ndarray, *,
                    eps_km: float, min_samples: int,
                    sample_size: int | None = None) -> ClusteringQuality:
    """The four checks that replace accuracy for an unsupervised model.

    Silhouette (same metric as TP5), the noise share, the cluster sizes, and -
    filled in by the caller - reproducibility across two runs.
    """

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    assigned = labels != -1
    noise_share = float(1 - assigned.mean()) if len(labels) else 1.0

    silhouette = None
    if n_clusters >= 2 and assigned.sum() > n_clusters:
        # The silhouette is O(n^2) in pairwise distances. On a grid of ~80 cells
        # that is free; on thousands of raw occurrence points it dominates a
        # 28-combination scan, so the species path passes a sample size. The
        # random_state is fixed, because a score that changes between two
        # identical runs would break the reproducibility check.
        subsample = (sample_size if sample_size and assigned.sum() > sample_size
                     else None)
        silhouette = float(silhouette_score(
            _radians(cells)[assigned], labels[assigned], metric="haversine",
            sample_size=subsample,
            random_state=0 if subsample else None))

    sizes = (pd.Series(labels[assigned]).value_counts().to_dict()
             if assigned.any() else {})

    return ClusteringQuality(
        n_clusters=int(n_clusters),
        noise_share=round(noise_share, 3),
        silhouette=round(silhouette, 3) if silhouette is not None else None,
        cluster_sizes={int(k): int(v) for k, v in sizes.items()},
        eps_km=eps_km,
        min_samples=min_samples,
    )


def tune(cells: pd.DataFrame, fallback_eps: float, fallback_min_samples: int,
         eps_grid: list[float] | None = None
         ) -> tuple[float, int, list[dict], bool]:
    """Scan eps x min_samples and choose the best admissible configuration.

    GridSearchCV cannot be used: it needs ground-truth labels to score, and
    clustering has none. This explicit scan over 7 x 4 combinations replaces it,
    scoring each on silhouette, noise share and cluster count.

    A configuration is admissible only with at least 2 clusters and a noise share
    inside the documented band - 0% means nothing was rejected, 100% means
    nothing was found. If none qualifies, the documented defaults are kept and
    ``tuned`` comes back False, so the notebook never silently invents settings.

    ``eps_grid`` overrides the documented candidates. It has to be overridable:
    the documented grid runs from 60 to 250 km, and on a 27 km island every one
    of those values puts every cell in one cluster - the scan then "finds
    nothing" for a reason that has nothing to do with the data.
    """

    low, high = M3_NOISE_BAND
    rows: list[dict] = []

    for eps_km in (eps_grid or M3_EPS_GRID):
        for min_samples in M3_MIN_SAMPLES_GRID:
            if len(cells) <= min_samples:
                continue
            _, labels = run_dbscan(cells, eps_km, min_samples)
            quality = measure_quality(cells, labels, eps_km=eps_km,
                                      min_samples=min_samples)
            rows.append({
                "eps_km": eps_km,
                "min_samples": min_samples,
                "n_clusters": quality.n_clusters,
                "noise_share": quality.noise_share,
                "silhouette": quality.silhouette,
                "admissible": (quality.n_clusters >= 2
                               and low < quality.noise_share < high
                               and quality.silhouette is not None),
            })

    admissible = [r for r in rows if r["admissible"]]
    if not admissible:
        return fallback_eps, fallback_min_samples, rows, False

    best = max(admissible, key=lambda r: r["silhouette"])
    return float(best["eps_km"]), int(best["min_samples"]), rows, True
