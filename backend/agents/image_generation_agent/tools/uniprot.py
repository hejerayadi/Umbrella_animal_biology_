"""Resolve a protein from gene and species via UniProt REST API."""

from __future__ import annotations

import logging
from typing import Any
from urllib.parse import quote

import requests

from ..tool_schemas import UniProtResult

logger = logging.getLogger(__name__)

UNIPROT_BASE_URL = "https://rest.uniprot.org"
DEFAULT_TIMEOUT = 30

REVIEWED = "UniProtKB reviewed (Swiss-Prot)"


def call_uniprot(
    gene: str,
    species: str | None = None,
    *,
    session: requests.Session | None = None,
    timeout: int = DEFAULT_TIMEOUT,
) -> UniProtResult:
    """Look up a protein by gene symbol and optional species name."""
    gene = gene.strip()
    if not gene:
        return UniProtResult(success=False, error="Gene symbol is required.")

    species = species.strip() if species else None
    http = session or requests.Session()

    try:
        entries = _search_uniprot(http, gene, species, timeout)
    except requests.RequestException as exc:
        logger.warning("UniProt search failed: %s", type(exc).__name__)
        return UniProtResult(success=False, error=f"UniProt request failed: {exc}")

    if not entries:
        detail = f" for gene '{gene}'"
        if species:
            detail += f" in '{species}'"
        return UniProtResult(success=False, error=f"No UniProt entry found{detail}.")

    entry = _select_best_entry(entries)
    parsed = _parse_entry(entry)
    if parsed.success and _entry_is_complete(parsed):
        return parsed

    accession = entry.get("primaryAccession") or entry.get("uniProtkbId") or parsed.accession
    if not accession:
        return UniProtResult(success=False, error="UniProt entry lacks an accession.")

    try:
        full_entry = _fetch_entry(http, str(accession), timeout)
    except requests.RequestException as exc:
        logger.warning("UniProt entry fetch failed: %s", type(exc).__name__)
        if parsed.success:
            return parsed
        return UniProtResult(success=False, error=f"UniProt entry fetch failed: {exc}")

    return _parse_entry(full_entry)


def _search_uniprot(
    session: requests.Session,
    gene: str,
    species: str | None,
    timeout: int,
) -> list[dict[str, Any]]:
    terms = [f"(gene:{gene}) OR (gene_exact:{gene})"]
    if species:
        terms.append(f'(organism_name:"{species}")')
    params = {
        "query": " AND ".join(terms),
        "format": "json",
        "size": 10,
    }
    response = session.get(
        f"{UNIPROT_BASE_URL}/uniprotkb/search",
        params=params,
        timeout=timeout,
    )
    response.raise_for_status()
    payload = response.json()
    results = payload.get("results")
    return list(results) if isinstance(results, list) else []


def _fetch_entry(
    session: requests.Session,
    accession: str,
    timeout: int,
) -> dict[str, Any]:
    response = session.get(
        f"{UNIPROT_BASE_URL}/uniprotkb/{quote(accession)}.json",
        timeout=timeout,
    )
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise ValueError("UniProt returned an unexpected payload shape.")
    return payload


def _select_best_entry(entries: list[dict[str, Any]]) -> dict[str, Any]:
    reviewed = [entry for entry in entries if entry.get("entryType") == REVIEWED]
    pool = reviewed or entries
    return pool[0]


def _parse_entry(entry: dict[str, Any]) -> UniProtResult:
    accession = entry.get("primaryAccession") or entry.get("uniProtkbId")
    if not accession:
        return UniProtResult(success=False, error="UniProt entry lacks an accession.")

    protein_name = _extract_protein_name(entry)
    organism = _extract_organism(entry)
    sequence = _extract_sequence(entry)
    sequence_length = _extract_sequence_length(entry, sequence)

    return UniProtResult(
        success=True,
        accession=str(accession),
        protein_name=protein_name,
        organism=organism,
        sequence_length=sequence_length,
        sequence=sequence,
    )


def _extract_protein_name(entry: dict[str, Any]) -> str | None:
    description = entry.get("proteinDescription") or {}
    recommended = description.get("recommendedName") or {}
    full_name = recommended.get("fullName") or {}
    if isinstance(full_name, dict):
        value = full_name.get("value")
        if isinstance(value, str) and value.strip():
            return value.strip()

    submission = description.get("submissionNames") or []
    if submission:
        first = submission[0] or {}
        full = (first.get("fullName") or {}).get("value")
        if isinstance(full, str) and full.strip():
            return full.strip()
    return None


def _extract_organism(entry: dict[str, Any]) -> str | None:
    organism = entry.get("organism") or {}
    scientific = organism.get("scientificName")
    if isinstance(scientific, str) and scientific.strip():
        return scientific.strip()
    return None


def _extract_sequence(entry: dict[str, Any]) -> str | None:
    sequence = entry.get("sequence") or {}
    value = sequence.get("value")
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _extract_sequence_length(entry: dict[str, Any], sequence: str | None) -> int | None:
    sequence_block = entry.get("sequence") or {}
    length = sequence_block.get("length")
    if isinstance(length, int) and length > 0:
        return length
    if sequence:
        return len(sequence)
    return None


def _entry_is_complete(result: UniProtResult) -> bool:
    return bool(
        result.accession
        and result.protein_name
        and result.organism
        and result.sequence_length
        and result.sequence
    )
