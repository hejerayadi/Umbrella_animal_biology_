"""Input and output data classes for M3 - Biodiversity Hotspots.

Implements section 7.2 of ``Biodiversity Hotspots.pdf``. Types only: no
clustering logic lives in this file.

Two deviations from the document, both deliberate and both reported in the
output warnings rather than hidden:

* ``cell_size_km`` replaces ``grid_resolution`` (H3), because equal-area
  sinusoidal squares stand in for H3 hexagons here. The areas are equal; only
  the cell shape differs.
* ``HotspotValidation`` is kept in the contract but returned unpopulated: the
  CEPF shapefile needed for the overlap check is not loaded.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


# ----------------------------------------------------------------- enums


class HotspotIndex(str, Enum):
    """Which measure ranks the cells. Selectable, because the four answer
    different questions and can order the same region differently."""

    RICHNESS = "richness"                  # distinct species per cell
    SHANNON = "shannon"                    # diversity with evenness
    SIMPSON = "simpson"                    # chance two records differ
    THREAT_WEIGHTED = "threat_weighted"     # IUCN-weighted - needs a Red List token


class ClusterAlgorithm(str, Enum):
    DBSCAN = "dbscan"       # the default, doc §4.4
    HDBSCAN = "hdbscan"     # retained as the documented Sprint 3 alternative


# ----------------------------------------------------------------- input


@dataclass
class BiodiversityHotspotInput:
    """Exactly one of ``region_name`` or ``bbox`` must be supplied.

    If both are present, ``bbox`` wins - it is the more specific instruction.
    """

    region_name: str | None = None
    bbox: tuple[float, float, float, float] | None = None

    # --- scope
    taxon_filter: str = "Animalia"
    taxon_key: int = 1                      # 1 = Animalia in the GBIF backbone

    # Species mode. Set together (a name for the answer, a key for GBIF) and the
    # module switches from "where is biodiversity richest" to "where has this one
    # species been recorded" - a different question with a different ranking, so
    # the output states which one it answered. See pipeline/points.py.
    species_name: str | None = None
    species_key: int | None = None

    # Where the bounding box came from: one of the six documented regions, or a
    # gazetteer lookup. Reported, because "Yellowstone" resolving to a county in
    # Montana is a legitimate reading the reader has to be able to see.
    place_source: str = "design document"
    place_full_name: str | None = None

    # Set when the cell size was refined for a small study area, so the answer
    # can say the grid is not the documented 42 km and why.
    cell_size_note: str | None = None
    year_from: int | None = 1990
    year_to: int | None = 2025

    # --- gridding
    cell_size_km: float = 42.0
    min_records_per_cell: int = 10          # noise floor, applied first

    # --- measurement
    index: HotspotIndex = HotspotIndex.RICHNESS
    normalise_by_effort: bool = True        # doc §5.4 - default on

    # --- clustering
    algorithm: ClusterAlgorithm = ClusterAlgorithm.DBSCAN
    eps_km: float = 120.0
    min_samples: int = 5
    auto_tune: bool = True                  # run the documented scan over eps

    # --- output shaping
    top_n: int = 10
    include_grid: bool = True
    max_records: int = 9000

    def __post_init__(self) -> None:
        if not (self.region_name or self.bbox):
            raise ValueError("one of region_name or bbox is required")
        if self.cell_size_km <= 0:
            raise ValueError("cell_size_km must be positive")
        if self.eps_km <= 0 or self.min_samples < 2:
            raise ValueError("invalid DBSCAN parameters")


# ------------------------------------------------------- supporting types


@dataclass
class SpeciesCount:
    species_key: int
    scientific_name: str
    record_count: int
    iucn_category: str | None = None
    threat_weight: int = 0          # CR 4 ... LC 0; zero without a Red List token


@dataclass
class RichnessCell:
    """One equal-area cell. The atomic unit of the analysis."""

    cell_id: str
    center: tuple[float, float]     # lat, lon
    area_km2: float
    species_count: int              # richness - distinct species
    record_count: int               # sampling effort within scope
    shannon: float = 0.0
    simpson: float = 0.0
    chao1_estimate: float = 0.0
    corrected_richness: float = 0.0
    effort_factor: float = 1.0
    dominant_country: str | None = None
    cluster_id: int | None = None   # None = DBSCAN noise (label -1)
    threatened_count: int = 0


@dataclass
class HotspotCluster:
    cluster_id: int
    label: str                      # country-derived; see the naming caveat
    centroid: tuple[float, float]
    area_km2: float
    cell_count: int
    species_count: int
    record_count: int
    score: float                    # normalised 0-1 on the selected index
    rank: int
    mean_shannon: float = 0.0
    threatened_count: int = 0
    top_species: list[SpeciesCount] = field(default_factory=list)
    dominant_country: str | None = None
    dominant_ecoregion: str | None = None    # unset without the WWF layer
    cepf_overlap_percent: float = 0.0        # unset without the CEPF layer


@dataclass
class EffortCorrection:
    """Always returned, even when ``applied`` is False: the user must be able to
    see whether the correction was made. Doc §5.4."""

    method: str
    applied: bool
    median_effort: float = 0.0
    cells_downweighted: int = 0
    cells_upweighted: int = 0
    top5_changed: int = 0           # if the correction changes nothing, it is not working
    chao1_used: bool = True
    note: str = ""


@dataclass
class ClusteringQuality:
    """DBSCAN is unsupervised, so accuracy cannot be used. Doc §4 lists these
    four checks as the replacement."""

    n_clusters: int
    noise_share: float
    silhouette: float | None
    cluster_sizes: dict[int, int] = field(default_factory=dict)
    reproducible: bool | None = None
    eps_km: float = 0.0
    min_samples: int = 0
    tuned: bool = False
    scan: list[dict] = field(default_factory=list)   # the 28-combination scan


@dataclass
class HotspotValidation:
    """Independent check against the published CEPF hotspots. Doc §3.3.6.

    Present in the contract, unpopulated here: the shapefile is not loaded.
    """

    reference_layer: str = "CEPF 36 hotspots"
    available: bool = False
    clusters_matched: int = 0
    mean_overlap_percent: float = 0.0
    interpretation: str = "not computed - CEPF layer unavailable"


@dataclass
class LegendEntry:
    label: str
    colour_hex: str
    value: str | int | float | None = None


@dataclass
class MapRenderSpec:
    """M3 emits a render specification; the frontend draws it. Doc §2.6."""

    type: str = "heatmap"
    scale: str = "quantile"
    classes: int = 7
    bbox: tuple[float, float, float, float] | None = None
    center: tuple[float, float] | None = None
    zoom: int = 5
    basemap: str = "OpenStreetMap"
    legend: list[LegendEntry] = field(default_factory=list)
    html_path: str | None = None    # a rendered folium file, for the dashboard


@dataclass
class Citation:
    source: str
    identifier: str
    url: str = ""
    license: str = ""
    accessed_at: str = ""


# ----------------------------------------------------------------- output


@dataclass
class BiodiversityHotspotOutput:
    """Carried in ``AgentResult.output`` when the status is COMPLETED or PARTIAL."""

    region: str
    study_area_km2: float
    records_retrieved: int
    records_analysed: int
    species_analysed: int
    cells_analysed: int
    noise_cells: int
    clusters: list[HotspotCluster]
    ranked_hotspots: list[HotspotCluster]
    effort_correction: EffortCorrection
    quality: ClusteringQuality
    validation: HotspotValidation
    parameters_used: dict[str, Any]           # reproducibility, doc goal G5
    render_spec: MapRenderSpec
    citations: list[Citation]
    summary: str
    confidence: float
    cleaning_report: list[dict] = field(default_factory=list)
    grid: list[RichnessCell] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    # M3's own outcome (completed | partial), preserved after the result is
    # narrowed to the platform's four statuses. See workers/hotspots/status.py.
    status_detail: str = "completed"
