"""M3 step 7 - characterise and rank the clusters.

Doc §4.2 step 7. A cluster becomes a hotspot only once it carries the numbers a
conservation analyst needs to defend a budget: position, area, species count,
record count and a recognisable name.

Naming: the document joins the WWF terrestrial ecoregion layer and adopts the
ecoregion with the largest overlap, so the answer reads "Albertine Rift montane
forests" rather than "cluster 2". That shapefile is not loaded here, so the
dominant country code is used and the substitution is recorded in the output -
nobody should mistake it for an official ecoregion name.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..schema import HotspotCluster, HotspotIndex, SpeciesCount

def _score_for(index: HotspotIndex, cluster_cells: pd.DataFrame,
               species_count: int) -> float:
    """The quantity the chosen index ranks clusters on.

    Richness ranks on the cluster's total distinct species - that is what "the
    richest hotspot" means to the analyst reading the answer, and it is how the
    Sprint 3 results are reported. Shannon and Simpson are per-cell properties,
    so they are averaged across the cluster's cells instead. Threat-weighted
    falls back to richness while no IUCN token is configured, and the worker
    warns when it does.
    """

    if index is HotspotIndex.SHANNON:
        return float(cluster_cells["shannon"].mean())
    if index is HotspotIndex.SIMPSON:
        return float(cluster_cells["simpson"].mean())
    return float(species_count)


def build_clusters(
    cells: pd.DataFrame,
    occurrences: pd.DataFrame,
    *,
    index: HotspotIndex,
    top_n: int,
) -> tuple[list[HotspotCluster], list[HotspotCluster]]:
    """Return ``(all_clusters, ranked_top_n)``.

    Both are returned because the output contract carries both: ``clusters`` is
    the complete unranked set, ``ranked_hotspots`` the top N on the chosen index.
    """

    clustered = cells[cells["cluster_id"].notna()]
    if clustered.empty:
        return [], []

    clusters: list[HotspotCluster] = []

    for cluster_id, group in clustered.groupby("cluster_id"):
        cell_ids = set(group["cell_id"])
        records = occurrences[occurrences["cell_id"].isin(cell_ids)]

        top = (records.groupby(["speciesKey", "species"]).size()
               .sort_values(ascending=False).head(5))
        top_species = [
            SpeciesCount(species_key=int(key), scientific_name=str(name),
                         record_count=int(count))
            for (key, name), count in top.items()
        ]

        country = (records["countryCode"].mode().iat[0]
                   if records["countryCode"].notna().any() else None)

        species_count = int(records["speciesKey"].nunique())

        clusters.append(HotspotCluster(
            cluster_id=int(cluster_id),
            label=f"Hotspot {int(cluster_id)} ({country or '?'})",
            centroid=(round(float(group["lat"].mean()), 3),
                      round(float(group["lon"].mean()), 3)),
            area_km2=float(group["area_km2"].sum()),
            cell_count=int(len(group)),
            species_count=species_count,
            record_count=int(len(records)),
            score=_score_for(index, group, species_count),
            rank=0,                       # assigned below
            mean_shannon=round(float(group["shannon"].mean()), 2),
            top_species=top_species,
            dominant_country=country,
        ))

    # Rank on the selected index, then normalise score to 0-1 so the frontend can
    # colour clusters without knowing which index was chosen.
    clusters.sort(key=lambda c: c.score, reverse=True)
    highest = clusters[0].score if clusters and clusters[0].score else 1.0
    for rank, cluster in enumerate(clusters, start=1):
        cluster.rank = rank
        cluster.score = round(float(cluster.score / highest), 3) if highest else 0.0

    return clusters, clusters[:top_n]


def quantile_breaks(values: np.ndarray, classes: int = 7) -> list[float]:
    """Quantile class breaks for the heatmap legend (doc render_spec scale)."""

    if len(values) == 0:
        return []
    quantiles = np.linspace(0, 100, classes + 1)
    return [round(float(v), 2) for v in np.percentile(values, quantiles)]
