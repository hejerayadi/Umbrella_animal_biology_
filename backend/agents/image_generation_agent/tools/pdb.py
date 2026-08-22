"""Search and rank PDB structures for a UniProt accession."""

from __future__ import annotations

import logging
from typing import Any

import requests

from ..tool_schemas import PDBStructureResult

logger = logging.getLogger(__name__)

RCSB_SEARCH_URL = "https://search.rcsb.org/rcsbsearch/v2/query"
RCSB_DATA_BASE_URL = "https://data.rcsb.org/rest/v1/core"
DEFAULT_TIMEOUT = 30

METHOD_PRIORITY = {
    "X-RAY DIFFRACTION": 1.0,
    "ELECTRON MICROSCOPY": 0.85,
    "SOLUTION NMR": 0.75,
    "SOLID-STATE NMR": 0.7,
    "NEUTRON DIFFRACTION": 0.65,
    "ELECTRON CRYSTALLOGRAPHY": 0.6,
}


def call_pdb(
    uniprot_accession: str,
    sequence_length: int | None = None,
    *,
    session: requests.Session | None = None,
    timeout: int = DEFAULT_TIMEOUT,
    max_candidates: int = 10,
) -> PDBStructureResult:
    """Return the best experimental structure for a UniProt accession."""
    accession = uniprot_accession.strip().upper()
    if not accession:
        return PDBStructureResult(
            found=False,
            message="UniProt accession is required for PDB lookup.",
        )

    http = session or requests.Session()

    try:
        entity_refs = _search_by_uniprot(http, accession, timeout, max_candidates)
    except requests.RequestException as exc:
        logger.warning("RCSB search failed: %s", type(exc).__name__)
        return PDBStructureResult(
            found=False,
            message=f"RCSB PDB search failed: {exc}",
        )

    if not entity_refs:
        return PDBStructureResult(
            found=False,
            message="No suitable structure found for the UniProt accession.",
        )

    candidates: list[tuple[float, dict[str, Any]]] = []
    for reference in entity_refs:
        try:
            metadata = _load_entity_metadata(http, reference, accession, sequence_length, timeout)
        except requests.RequestException as exc:
            logger.warning("RCSB metadata fetch failed for %s: %s", reference, type(exc).__name__)
            continue
        if metadata is not None:
            candidates.append((_score_candidate(metadata), metadata))

    if not candidates:
        return PDBStructureResult(
            found=False,
            message="No suitable structure found for the UniProt accession.",
        )

    candidates.sort(key=lambda item: item[0], reverse=True)
    best_score, best = candidates[0]
    return PDBStructureResult(
        found=True,
        pdb_id=best["pdb_id"],
        experimental_method=best["experimental_method"],
        resolution=best["resolution"],
        sequence_coverage=best["sequence_coverage"],
        title=best["title"],
        organism=best["organism"],
        chain=best["chain"],
        selection_reason=_selection_reason(best, best_score),
    )


def _search_by_uniprot(
    session: requests.Session,
    accession: str,
    timeout: int,
    max_candidates: int,
) -> list[str]:
    query = {
        "query": {
            "type": "group",
            "logical_operator": "and",
            "nodes": [
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": (
                            "rcsb_polymer_entity_container_identifiers."
                            "reference_sequence_identifiers.database_accession"
                        ),
                        "operator": "exact_match",
                        "value": accession,
                    },
                },
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": (
                            "rcsb_polymer_entity_container_identifiers."
                            "reference_sequence_identifiers.database_name"
                        ),
                        "operator": "exact_match",
                        "value": "UniProt",
                    },
                },
            ],
        },
        "return_type": "polymer_entity",
        "request_options": {
            "paginate": {"start": 0, "rows": max_candidates},
            "results_verbosity": "compact",
        },
    }
    response = session.post(RCSB_SEARCH_URL, json=query, timeout=timeout)
    response.raise_for_status()
    payload = response.json()
    result_set = payload.get("result_set") or []
    refs: list[str] = []
    for item in result_set:
        if isinstance(item, str):
            refs.append(item)
        elif isinstance(item, dict) and item.get("identifier"):
            refs.append(str(item["identifier"]))
    return refs


def _load_entity_metadata(
    session: requests.Session,
    reference: str,
    accession: str,
    sequence_length: int | None,
    timeout: int,
) -> dict[str, Any] | None:
    try:
        pdb_id, entity_id = reference.rsplit("_", 1)
    except ValueError:
        return None

    entity_response = session.get(
        f"{RCSB_DATA_BASE_URL}/polymer_entity/{pdb_id}/{entity_id}",
        timeout=timeout,
    )
    entry_response = session.get(
        f"{RCSB_DATA_BASE_URL}/entry/{pdb_id}",
        timeout=timeout,
    )
    entity_response.raise_for_status()
    entry_response.raise_for_status()
    entity = entity_response.json()
    entry = entry_response.json()

    identifiers = entity.get("rcsb_polymer_entity_container_identifiers") or {}
    references = identifiers.get("reference_sequence_identifiers") or []
    matching = [
        item
        for item in references
        if str(item.get("database_name", "")).lower() == "uniprot"
        and str(item.get("database_accession", "")).upper() == accession
    ]
    if not matching:
        return None

    coverage = _reference_coverage(entity, matching, sequence_length)
    auth_chains = identifiers.get("auth_asym_ids") or identifiers.get("asym_ids") or []
    chain = str(auth_chains[0]) if auth_chains else None

    resolution_values = (entry.get("rcsb_entry_info") or {}).get("resolution_combined") or []
    resolution = float(resolution_values[0]) if resolution_values else None
    method = ((entry.get("exptl") or [{}])[0]).get("method")
    title = (entry.get("struct") or {}).get("title")
    organism = _extract_organism(entity)

    return {
        "pdb_id": pdb_id.upper(),
        "experimental_method": method,
        "resolution": resolution,
        "sequence_coverage": coverage,
        "title": title,
        "organism": organism,
        "chain": chain,
    }


def _reference_coverage(
    entity: dict[str, Any],
    matching_references: list[dict[str, Any]],
    sequence_length: int | None,
) -> float:
    reported = [
        float(item["reference_sequence_coverage"])
        for item in matching_references
        if item.get("reference_sequence_coverage") is not None
    ]
    if reported:
        return max(0.0, min(1.0, max(reported)))

    entity_length = (entity.get("entity_poly") or {}).get("rcsb_sample_sequence_length")
    if entity_length and sequence_length:
        return max(0.0, min(1.0, float(entity_length) / float(sequence_length)))
    return 0.0


def _extract_organism(entity: dict[str, Any]) -> str | None:
    source = entity.get("rcsb_entity_source_organism") or []
    if source:
        scientific = source[0].get("ncbi_scientific_name")
        if isinstance(scientific, str) and scientific.strip():
            return scientific.strip()
    return None


def _score_candidate(metadata: dict[str, Any]) -> float:
    coverage = float(metadata.get("sequence_coverage") or 0.0)
    resolution = metadata.get("resolution")
    method = str(metadata.get("experimental_method") or "").upper()
    method_score = METHOD_PRIORITY.get(method, 0.5)
    resolution_score = max(0.0, 1.0 - ((float(resolution) if resolution is not None else 4.0) - 1.0) / 5.0)
    return round(0.5 * coverage + 0.3 * resolution_score + 0.2 * method_score, 4)


def _selection_reason(metadata: dict[str, Any], score: float) -> str:
    method = metadata.get("experimental_method") or "unknown method"
    resolution = metadata.get("resolution")
    coverage = metadata.get("sequence_coverage")
    resolution_text = f"{resolution:.2f} Å" if isinstance(resolution, (int, float)) else "unknown resolution"
    coverage_text = f"{coverage:.0%}" if isinstance(coverage, (int, float)) else "unknown coverage"
    return (
        f"Selected {metadata.get('pdb_id')} (score {score:.2f}) for {method}, "
        f"resolution {resolution_text}, sequence coverage {coverage_text}."
    )
